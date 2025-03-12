# check new_poses are correct

import numpy as np
from argparse import ArgumentParser
import json
import cv2
import open3d as o3d


flip_mat = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])


def depth_to_pointcloud(depth, rgb, intrinsic):
    """
    Convert depth image to a point cloud.

    Args:
        depth (np.ndarray): Depth image (H x W)
        rgb (np.ndarray): RGB image (H x W x 3)
        intrinsic (dict): Camera intrinsic parameters

    Returns:
        o3d.geometry.PointCloud: Generated point cloud
    """
    h, w = depth.shape
    fx, fy = intrinsic["fl_x"], intrinsic["fl_y"]
    cx, cy = intrinsic["cx"], intrinsic["cy"]

    # Generate grid of pixel coordinates
    i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')

    # Convert to 3D space (x, y, z)
    z = depth.astype(np.float32)
    x = (i - cx) * z / fx
    y = (j - cy) * z / fy

    # Stack to create point cloud
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    rgb = rgb.reshape(-1, 3)

    # Filter out invalid points (zero depth)
    valid = ((z > 0) & (z < 2.0)).reshape(-1)
    points = points[valid]
    rgb = rgb[valid]

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(rgb / 255.0)

    return pcd


def read_intrinsic(json_path):
    intrinsic_keys = ['fl_x', 'fl_y', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2']
    intrinsic_dict = {}
    
    with open(json_path, 'r') as fp:
        json_file = json.load(fp)

    for frame_idx in range(len(json_file['frames'])):
        frame = json_file['frames'][frame_idx]
        fname = frame['file_path']
        fname = fname.split('/')[-1]
        if 'color' in fname:
            img_idx = fname.replace('-color.png', '')
        else:
            img_idx = fname.replace('.png', '')

        if not str.isdigit(img_idx) or int(img_idx) > 50:
            for key in intrinsic_keys:
                intrinsic_dict[key] = frame[key]
            break

    return intrinsic_dict
            


def check_new_poses(new_pose_path, image_dir, json_path):
    new_poses = np.load(new_pose_path)
    intrinsic_dict = read_intrinsic(json_path)

    serial_numbers = new_poses.files

    pcds = []

    for serial_number in serial_numbers:
        pose = new_poses[serial_number]

        rgb_path = f"{image_dir}/{serial_number}.png"
        depth_path = f"{image_dir}/{serial_number}-depth.png"

        rgb_image = cv2.imread(rgb_path)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        depth_scale = 4000.0 if serial_number == "f1150952" else 1000.0
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) / depth_scale

        pcd = depth_to_pointcloud(depth_image, rgb_image, intrinsic_dict)

        pcds.append(pcd)

    # apply extrinsic transformation to point cloud
    for i, serial_number in enumerate(serial_numbers):
        pose = new_poses[serial_number]
        pose = pose @ flip_mat
        pcds[i].transform(pose)

    # add axis arrows
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    pcds.append(axis)
        
    o3d.visualization.draw_geometries(pcds)


    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--new_poses", default="./new_poses.npz")
    parser.add_argument("--image_dir", default="./images")
    parser.add_argument("--json_path", default="./transforms.json")

    args = parser.parse_args()
    
    check_new_poses(args.new_poses, args.image_dir, args.json_path)