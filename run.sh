python colmap_to_nerf.py --images images --run_colmap --overwrite --colmap_matcher exhaustive
python match_colmap_world.py --json_file transforms.json --txt_file cam_transforms.txt --visualize
python apply_matching_params_zed.py --json_file transforms.json --matching_params matching_params.npz  --visualize
python check_new_poses.py --image_dir ~/sangminkim/realsense/realsense