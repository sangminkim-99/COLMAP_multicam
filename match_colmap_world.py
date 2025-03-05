import os
import numpy as np
import argparse
import json
import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import sin, cos
from torch.autograd import Variable
from tqdm import tqdm
from chamferdist import ChamferDistance
import matplotlib.pyplot as plt


# Some global variables
flip_mat = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])
fix_pose_mat = np.array([
    [0, 1, 0, 0],
    [-1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])
# original value
ee2cam=[0.055, 0.0325, -0.05]
# torch found value
ee2cam=[0.0721,  0.0374, -0.0187]

def read_colmap(json_path):
    with open(json_path, 'r') as fp:
        json_file = json.load(fp)

    img_pose_dict = {}
    for frame_idx in range(len(json_file['frames'])):
        frame = json_file['frames'][frame_idx]
        fname = frame['file_path']
        fname = fname.split('/')[-1]
        if 'color' in fname:
            img_idx = int(fname.replace('-color.png', ''))
        else:
            img_idx = int(fname.replace('.png', ''))

        pose = np.array(frame['transform_matrix'])

        img_pose_dict[img_idx] = pose

    # remove poses that are not in txt file (for now, it is > 50)
    remove_keys = []
    for img_idx in img_pose_dict.keys():
        if img_idx > 50:
            remove_keys.append(img_idx)

    for key in remove_keys:
        del img_pose_dict[key]

    # now let's sort the pose by key
    poses = []
    for img_idx in range(len(img_pose_dict.keys())):
        poses.append(img_pose_dict[img_idx])
    
    return np.array(poses)


def read_world(args):
    ee_pose = np.loadtxt(args.txt_file)
    ee_pose = np.transpose(ee_pose.reshape(len(ee_pose), 4, 4), (0, 2, 1))
    # Change to cam pose
    cam_transforms = ee_pose.copy()
    cam_transforms[:, :3, 3] = ee_pose[:, :3, 3] + np.matmul(ee_pose[:, :3, :3], np.array(ee2cam))
    cam_transforms = cam_transforms.astype(np.float32)
    
    poses = []
    for frame_idx in range(len(cam_transforms)):
        cam_pose = cam_transforms[frame_idx]
        h_, w_ = cam_pose.shape
        if h_ == 3:
            last_row = np.array([0, 0, 0, 1]).reshape(1, 4)
            cam_pose = np.append(cam_pose, last_row, axis=0)
        # additional flipping for inmc made dataset
        cam_pose = np.matmul(cam_pose, fix_pose_mat)
        cam_pose = np.matmul(cam_pose, flip_mat)
        cam_pose = cam_pose.astype(np.float32)

        poses.append(cam_pose)

    return np.array(poses)


def align(model, data):
    """Align two trajectories using the method of Horn (closed-form).
    
    Input:
    model -- first trajectory (3xn)
    data -- second trajectory (3xn)
    
    Output:
    rot -- rotation matrix (3x3)
    trans -- translation vector (3x1)
    trans_error -- translational error per point (1xn)
    
    """
    # np.set_printoptions(precision=3,suppress=True)
    model_mean = model.mean(1)
    model_zerocentered = np.array([model[0] - model_mean[0], model[1] - model_mean[1], model[2] - model_mean[2]])
    data_mean = data.mean(1)
    data_zerocentered = np.array([data[0] - data_mean[0], data[1] - data_mean[1], data[2] - data_mean[2]])
    
    W = np.zeros( (3,3) )
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:,column],data_zerocentered[:,column])
    U,d,Vh = np.linalg.linalg.svd(W.transpose())
    S = np.matrix(np.identity( 3 ))
    if(np.linalg.det(U) * np.linalg.det(Vh)<0):
        S[2,2] = -1
    rot = U*S*Vh

    rotmodel = rot*model_zerocentered
    dots = 0.0
    norms = 0.0

    for column in range(data_zerocentered.shape[1]):
        dots += np.dot(data_zerocentered[:,column].transpose(),rotmodel[:,column])
        normi = np.linalg.norm(model_zerocentered[:,column])
        norms += normi*normi

    s = float(dots/norms)    

    print("scale: %f " % s)  
    
    trans = data.mean(1).reshape(3, 1) - s*rot * model.mean(1).reshape(3, 1)

    model_aligned = s*rot * model + trans
    alignment_error = model_aligned - data
    
    trans_error = np.sqrt(np.sum(np.multiply(alignment_error,alignment_error),0)).A[0]

    return rot,trans,trans_error, s, model_aligned, alignment_error, model_mean


def align_torch(model, data_, args, optimize_ee2cam=False):
    device = "cuda"
    data_ = torch.from_numpy(data_).float().to(device)

    ee_pose = np.loadtxt(args.txt_file)
    ee_pose = np.transpose(ee_pose.reshape(len(ee_pose), 4, 4), (0, 2, 1))
    ee_pose = torch.from_numpy(ee_pose).float()
    model = torch.from_numpy(model).to(device).float()
    mseloss = torch.nn.MSELoss()
    # data = torch.from_numpy(data).to(device).float()

    # optimization variables
    roll = Variable(torch.tensor([0]).float().to(device), requires_grad=True)
    pitch = Variable(torch.tensor([0]).float().to(device), requires_grad=True)
    yaw = Variable(torch.tensor([0]).float().to(device), requires_grad=True)
    scale = Variable(torch.tensor([1]).float().to(device), requires_grad=True)
    trans = Variable(torch.zeros(3, 1).float().to(device), requires_grad=True)
    ee2cam = Variable(torch.tensor([0.055, 0.0325, -0.05]).float().to(device), requires_grad=True)
    optimizer = optim.Adam([
            {'params': trans, 'lr': 0.01},
            {'params': roll, 'lr': 0.01},
            {'params': pitch, 'lr': 0.01},
            {'params': yaw, 'lr': 0.01},
            {'params': scale, 'lr': 0.01},
            {'params': ee2cam, 'lr': 0.001},
        ])
    scaler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.9)
    chamfer = ChamferDistance()

    def rpy_to_rotmat(roll, pitch, yaw):
        # rotation matrix from roll, pitch, yaw
        tensor_0 = torch.zeros(1, device=device)
        tensor_1 = torch.ones(1, device=device)
        RX = torch.stack([
            torch.stack([tensor_1, tensor_0, tensor_0]),
            torch.stack([tensor_0, cos(roll), -sin(roll)]),
            torch.stack([tensor_0, sin(roll), cos(roll)])]).reshape(3, 3)

        RY = torch.stack([
                        torch.stack([cos(pitch), tensor_0, sin(pitch)]),
                        torch.stack([tensor_0, tensor_1, tensor_0]),
                        torch.stack([-sin(pitch), tensor_0, cos(pitch)])]).reshape(3, 3)

        RZ = torch.stack([
                        torch.stack([cos(yaw), -sin(yaw), tensor_0]),
                        torch.stack([sin(yaw), cos(yaw), tensor_0]),
                        torch.stack([tensor_0, tensor_0, tensor_1])]).reshape(3, 3)

        r = torch.mm(RZ, RY)
        r = torch.mm(r, RX)
        return r

    _, N = model.shape
    # optimize
    for i_ter in tqdm(range(2000)):
        optimizer.zero_grad()
        # calculate loss
        rot_mat = rpy_to_rotmat(roll, pitch, yaw)
        model_mean = model.mean(dim=1).view(3, 1) # (3, 1)
        transformed_model = model - model_mean
        transformed_model = rot_mat @ transformed_model
        transformed_model = scale * transformed_model
        transformed_model = transformed_model + model_mean.repeat((1, N)) + trans.view(3, 1)
        # loss = chamfer(transformed_model.unsqueeze(dim=0), data.unsqueeze(dim=0), bidirectional=True, point_reduction='mean')
        # let's get our data
        if optimize_ee2cam:
            cam_transforms = ee_pose.to(device)
            cam_transforms[:, :3, 3] = cam_transforms[:, :3, 3] + (cam_transforms[:, :3, :3] @ (ee2cam))
            data = cam_transforms @ torch.from_numpy(fix_pose_mat).float().to(device)
            data = data @ torch.from_numpy(flip_mat).float().to(device)
            data = data[:, :3, -1]
            data = data.transpose(0, 1)
            diff = data - transformed_model
        else:
            diff = data_ - transformed_model
        dist = torch.sqrt(torch.sum(diff ** 2, dim=0))
        loss = dist.mean()
        loss.backward()
        optimizer.step()
        scaler.step(loss)

    # visualize the final result
    if optimize_ee2cam:
        data = data.detach().cpu().numpy()
    else:
        data = data_.detach().cpu().numpy()
    transformed_model = transformed_model.detach().cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for i in range(N):
        ax.scatter(data[0], data[1], data[2], c='r')
        ax.scatter(transformed_model[0], transformed_model[1], transformed_model[2], c='c')
    plt.show()

    print("For torch based optim:")
    print("Final loss: {}".format(loss.item()))
    print("ee2cam:")
    print(ee2cam.detach().cpu().numpy())

    # return the output
    # return rot_mat.detach().cpu().numpy(), model_mean.cpu().numpy(), scale.detach().cpu().numpy(), trans.detach().cpu().numpy()
 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", default="/home/twjhlee/Data_ssd/INMC/241002/241002_colmap_side/transforms_train.json")
    parser.add_argument("--txt_file", default="/home/twjhlee/Data_ssd/INMC/241002/241002_colmap_side/cam_transforms.txt")
    parser.add_argument("--visualize", action='store_true')
    args = parser.parse_args()

    # read world poses
    world_poses = read_world(args)
    # colmap poses
    colmap_poses =  read_colmap(args.json_file)

    # Let's start aligning the poses
    world_points = world_poses[:, :3, -1]
    colmap_points = colmap_poses[:, :3, -1]

    rot, trans, trans_error, s, model_aligned, alignment_error, mean_pos = align(colmap_points.transpose(), world_points.transpose())
    print("Final trans error for horn's method: {}".format(trans_error.mean()))
    align_torch(colmap_points.transpose(), world_points.transpose(), args)
    

    # final_poses
    final_poses = colmap_poses.copy()
    for idx in range(len(final_poses)):
        rotation = np.matmul(rot, final_poses[idx, :3, :3])
        translation = model_aligned[:, idx]
        final_poses[idx, :3, :3] = rotation
        final_poses[idx, 0, -1] = translation[0]
        final_poses[idx, 1, -1] = translation[1]
        final_poses[idx, 2, -1] = translation[2]

    # save to disk
    rot = np.array(rot)
    trans = np.array(trans)
    s = np.array([s])

    np.savez_compressed("matching_params.npz", rotation=rot, translation=trans, scale=s, mean_pos=mean_pos)

