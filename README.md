# DeepLLE

Distributed PyTorch Training Framework for Low-light Image/Video Enhancement.


## Installation

### Requirements
- Linux or macOS with Python ≥ 3.7
- PyTorch ≥ 1.8 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
- OpenCV

### Build DeepLLE from source

cd to the root directory of the project and then install using pip:
```
python -m pip install -e .
```

## Getting Started

### Training
In `tools`, we provide a basic training script `train_net.py`. You can use it as a reference to write your own training scripts.  
To train with `tools/train_net.py`, you can run:
```
cd tools/
python train_net.py --num-gpus 4 --config ../configs/llie_base.json
```
The config is made to train with 4 GPUs. You can change the number of GPUs by modifying the `--num-gpus` option.

To specify the GPU devices, you can setting the environment variable `CUDA_VISIBLE_DEVICES`:
```
CUDA_VISIBLE_DEVICES=0,2 python train_net.py --num-gpus 2 --config ../configs/llie_base.json
```
The above config will use the GPU devices with id 0 and 2 for training.
