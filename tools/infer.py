# Created on Sat Oct 08 2022 by Chuyang Zhao
import argparse
import os
import sys
import torch
import json
from typing import Optional, Any
import numpy as np
import cv2
import tqdm

from deeplle.engine.launch import launch
from deeplle.utils import init_config, mkdirs, check_path_exists, check_path_is_image, comm
from deeplle.utils.checkpoint import Checkpointer
from deeplle.utils.logger import setup_logger
from deeplle.utils.image_ops import save_image
from deeplle.utils.nn_utils import get_bare_model
from deeplle.modeling import build_model, create_ddp_model
from deeplle.data import build_transforms, build_test_loader, CommISPDataset


def default_argument_parser(epilog=None):
    """
    Create a parser with some common arguments used by detectron2 users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog or f"""
Examples:

Run on single machine:
    $ {sys.argv[0]} --num-gpus 8 --config configs/base.json

Change some config options:
    $ {sys.argv[0]} --config configs/base.json --opts model.name=ResNet50 solver.optimizer.args.lr=0.001
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2**15 + 2**14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "--opts",
        help='modify config using the command line, e.g. --opts model.name "ResNet50" data.batch_size=30',
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def build_test_model(cfg_model):
    # turn on testing when in testing mode
    cfg_model["args"]["testing"] = True

    model = build_model(cfg_model)
    model = create_ddp_model(model)
    
    return model


def resume_checkpoint(model: Any, path: str, ema_model: bool = False):
    """
    Args:
        model (nn.Module): model to be resumed.
        path (str): path to checkpoint file.
        ema_model (bool): whether to resume the ema model.
    """
    model = get_bare_model(model)
    Checkpointer.resume_checkpoint(model=model, resume_path=path, ema_model=ema_model)


def build_data_loader(config):
    cfg_infer_factory = config["infer"]["data"]
    transforms = build_transforms(cfg_infer_factory["transforms"])

    dataset_dicts = load_data(cfg_infer_factory["img_dir"], config["infer"]["save_dir"])
    dataset = CommISPDataset(dataset_dicts, False, transforms)
    dataloader = build_test_loader(dataset=dataset, batch_size=cfg_infer_factory["batch_size"], num_workers=cfg_infer_factory["num_workers"])
    return dataloader


def read_images_from_dir(img_dir):
    """
    Read all images in img_dir.

    Deprecated: use load_data instead.
    """
    dataset_dicts = []
    for filename in os.listdir(img_dir):
        path = os.path.join(img_dir, filename)
        if not check_path_is_image(path): continue
        record = {'image_path': path}
        dataset_dicts.append(record)
    return dataset_dicts


def load_data(img_dir: str, save_dir: str):
    """
    Load all images in img_dir recursively and create corresponding 
    directories rooted in the save_dir if any image is found.

    Args:
        img_dir (str): root directoy path of the input images.
        save_dir (str): root directory of the saved images.
    """
    dataset_dicts = []
    for root, _, files in os.walk(img_dir):
        for filename in files:
            if check_path_is_image(filename):
                relative_dir = root[len(img_dir):]
                if relative_dir.startswith("/"): relative_dir = relative_dir[1:]
                output_dir = os.path.join(save_dir, relative_dir)
                
                if not check_path_exists(output_dir):
                    mkdirs(output_dir)
                
                input_path = os.path.join(root, filename)
                output_path = os.path.join(output_dir, filename)
                record = {'image_path': input_path, "save_path": output_path}
                dataset_dicts.append(record)
    
    return dataset_dicts


def post_process(image: np.array, maxrange: float=0.8, highpercent: int=95, lowpercent: int=5, hsvgamma: float=0.8):
    """
    Post process procedure used in MBLLEN.

    Args:
        image (np.ndarray): numpy image, data range is in [0, 1].
    """
    gray_image = image[:, :, 0] * 0.299 + image[:, :, 1] * 0.587 + image[:, :, 1] * 0.114
    percent_max = sum(sum(gray_image >= maxrange))/sum(sum(gray_image <= 1.0))
    # print(percent_max)
    max_value = np.percentile(gray_image[:], highpercent)
    if percent_max < (100-highpercent)/100.:
        scale = maxrange / max_value
        image = image * scale
        image = np.minimum(image, 1.0)

    gray_image = image[:,:,0]*0.299 + image[:,:,1]*0.587 + image[:,:,1]*0.114
    sub_value = np.percentile(gray_image[:], lowpercent)
    image = (image - sub_value)*(1./(1-sub_value))

    imgHSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(imgHSV)
    S = np.power(S, hsvgamma)
    imgHSV = cv2.merge([H, S, V])
    image = cv2.cvtColor(imgHSV, cv2.COLOR_HSV2RGB)
    image = np.clip(image, 0.0, 1.0)
    return image


@torch.no_grad()
def infer(model, dataloader, save_pair: bool = False):
    """
    Inference function.

    Args:
        model (nn.Module): model to be tested.
        dataloader (DataLoader): dataloader for inference.
    """
    if comm.is_main_process():
        indices = tqdm.tqdm(dataloader, total=len(dataloader))
    else:
        indices = dataloader
    
    for data in indices:
        outputs = model(data)

        for output, record in zip(outputs, data):
            save_path = record["save_path"]
            input_image = record["image"] if save_pair else None
            save_image(save_path, output, input_image=input_image)


def main(args):
    # initialize config
    config = init_config(args)

    # create the test output directory if it does not exist.
    mkdirs(config["infer"]["save_dir"])

    # setup logger
    rank = comm.get_rank()
    setup_logger(output=config["trainer"]["log_dir"], distributed_rank=rank, name="deeplle")
    logger = setup_logger(output=config["trainer"]["log_dir"], distributed_rank=rank, name="deeplle.inference")

    logger.info(f"Configuration:\n{json.dumps(config, indent=4)}")

    # build model
    model = build_test_model(config["model"])
    resume_checkpoint(model, config["infer"]["resume_checkpoint"])

    # build dataloader
    dataloader = build_data_loader(config)

    # do inference and save the results
    infer(model, dataloader, save_pair=config["infer"]["save_pair"])


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
