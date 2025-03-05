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

    # remove poses that are robot poses (for now, it is < 50)
    remove_keys = []
    for img_idx in img_pose_dict.keys():
        if img_idx < 50:
            remove_keys.append(img_idx)

    for key in remove_keys:
        del img_pose_dict[key]

    # convert all serial numbers to string
    img_pose_dict = {str(k): v for k, v in img_pose_dict.items()}
    
    return img_pose_dict # key: serial number, value: pose matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", default="./transforms.json")
    parser.add_argument("--matching_params", default="./matching_params.npz")
    parser.add_argument("--visualize", action='store_true')
    args = parser.parse_args()

    # colmap poses
    colmap_poses =  read_colmap(args.json_file)

    matching_params = np.load(args.matching_params)
    
    rotation = matching_params['rotation'] # (3x3) 
    translation = matching_params['translation'].squeeze() # (3,)
    scale = matching_params['scale'] # (1)
    mean_pos = matching_params['mean_pos'].squeeze() # (3,)

    # apply matching params to colmap poses
    new_poses = {}
    for img_idx, pose in colmap_poses.items():
        new_pose = pose.copy()
        new_pose[:3, :3] = rotation @ pose[:3, :3]
        new_pose[:3, 3] = scale * (rotation @ pose[:3, 3]) + translation
        new_poses[img_idx] = new_pose
        

    # visualize the position of the poses
    if args.visualize:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for img_idx, pose in new_poses.items():
            ax.scatter(pose[0, 3], pose[1, 3], pose[2, 3])
        # axis limits
        ax.set_xlim(0, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(0, 1)
        plt.show()

    # save new_poses dict
    np.savez('new_poses.npz', **new_poses)