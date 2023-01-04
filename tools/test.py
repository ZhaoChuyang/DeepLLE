# Created on Sat Oct 08 2022 by Chuyang Zhao
import argparse
import os
import sys
import torch
from torch import nn
from collections import defaultdict
import json
from typing import Dict, List, Callable
import numpy as np
import cv2
import tqdm

from deeplle.utils import init_config, resume_checkpoint
from deeplle.utils import image_ops, comm
from deeplle.utils.logger import setup_logger
from deeplle.engine.launch import launch
from deeplle.modeling import build_model, create_ddp_model
from deeplle.modeling.metrics import build_metric
from deeplle.data import build_transforms, build_test_loader


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


def build_data_loader(cfg_test_factory):
    transforms = build_transforms(cfg_test_factory["transforms"])

    dataloader = build_test_loader(
        names=cfg_test_factory["names"],
        batch_size=cfg_test_factory["batch_size"],
        num_workers=cfg_test_factory["num_workers"],
        transforms=transforms
    )

    return dataloader


def build_test_metrics(cfg_metrics: List) -> Dict:
    metrics = {}

    for cfg in cfg_metrics:
        if isinstance(cfg, str):
            name = cfg
            args = {}
        else:
            name = cfg["name"]
            args = cfg["args"]
        metric_fn = build_metric(name, args)
        metrics[name] = metric_fn

    return metrics


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
def test(dataloader, model, metrics: List[Dict[str, Callable]]):
    results = defaultdict(list)

    if comm.is_main_process():
        indices = tqdm.tqdm(dataloader, total=len(dataloader))
    else:
        indices = dataloader
    
    for data in indices:
        outputs = model(data)
        for output, record in zip(outputs, data):
            target = record["target"]
            target = image_ops.convert_to_image(target)
            output = image_ops.convert_to_image(output)

            for metric_name, metric_fn in metrics.items():
                results[metric_name].append(metric_fn(output, target))
    
    return results


def main(args):
    config = init_config(args)

    rank = comm.get_rank()
    setup_logger(output=config["trainer"]["log_dir"], distributed_rank=rank, name="deeplle")
    logger = setup_logger(output=config["trainer"]["log_dir"], distributed_rank=rank, name="deeplle.testing")

    logger.info(f"Configuration:\n{json.dumps(config, indent=4)}")

    model = build_test_model(config["model"])
    resume_checkpoint(model, config["test"]["resume_checkpoint"])

    metrics = build_test_metrics(config["test"]["metrics"])

    # build test dataloader
    cfg_test_factory = config["data_factory"]["test"]
    logger.info(f"=====> Testing on {cfg_test_factory['names']} datasets")
    dataloader = build_data_loader(cfg_test_factory)

    results = test(dataloader, model, metrics)
    all_results = comm.gather(results)

    if comm.is_main_process():
        results = {k: np.mean([r[k] for r in all_results], axis=0) for k in all_results[0].keys()}
        for metric_name, records in results.items():
            logger.info(f"{metric_name} : {sum(records) / len(records)}")


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
