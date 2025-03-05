# COLMAP to Real-World Pose Conversion

It is hard to find extrinsic matrices for external cameras in the real world, especially when you have a fixed world-coordinate frame. This repository provides a way to convert the extrinsic matrices from COLMAP to real-world poses.

## Installation
1. Clone this repository to your local machine.
2. `conda create -n colmap_to_real python=3.9`
3. Install Pytorch according to your system from [here](https://pytorch.org/get-started/locally/).
4. Install the required dependencies by running `pip install -r requirements.txt`.

## Usage
1. Prepare your dataset under images, put the real-world poses with cam_transforms.txt
2. Run the script with the following command:
```bash
bash run.sh # here, you should check all paths are correct
```

# Results

## Registration of RGB-D images

# Acknowledgements

- `colmap_to_nerf.py` is adapted from [instant-ngp](https://github.com/NVlabs/instant-ngp/blob/master/scripts/colmap2nerf.py).
- `match_colmap_world.py` is adapted from @[Junho Lee](https://github.com/twjhlee)'s [code](#).