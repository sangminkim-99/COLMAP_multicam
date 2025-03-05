# COLMAP to Real-World Pose Conversion

## Installation
1. Clone this repository to your local machine.
2. conda create -n colmap_to_real python=3.9
3. Install Pytorch according to your system from [here](https://pytorch.org/get-started/locally/).
4. Install the required dependencies by running `pip install -r requirements.txt`.

## Usage
1. Prepare your dataset under images, put the real-world poses with cam_transforms.txt
2. Run the script with the following command:
```bash
bash run.sh # here, you should check all paths are correct
```
