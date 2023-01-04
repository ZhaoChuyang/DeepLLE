# Created on Sat Oct 08 2022 by Chuyang Zhao
import argparse
import os
import torch
from torch import nn
from collections import defaultdict
import json
from typing import Dict, List, Optional, Callable, Union
import numpy as np
import cv2
import tqdm

from deeplle.utils import init_config, check_path_exists, check_path_is_image
from deeplle.modeling import build_model
from deeplle.modeling.metrics import build_metric
from deeplle.data import build_transforms, build_test_loader, CommISPDataset


def build_test_model(cfg_model, device):
    # turn on testing when in testing mode
    cfg_model["args"]["testing"] = True

    model = build_model(cfg_model)
    model = nn.parallel.DataParallel(model)
    model.to(device)
    
    return model


def resume_checkpoint(model, resume_path):
    if not check_path_exists(resume_path):
        raise FileNotFoundError("Checkpoint to resume was not found in {}".format(resume_path))
        
    print("Loading checkpoint: {}...".format(resume_path))
    checkpoint = torch.load(resume_path)

    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['state_dict'])
    
    print("Model's state dict loaded, missing keys: {}, unexpected keys: {}".format(missing_keys, unexpected_keys))


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


def save_image(save_path: str, image: torch.Tensor, do_post_process: bool=False, input_image: Optional[torch.Tensor] = None):
    """
    Save image to the specified path. If input image is provided, construct image pair
    by concatenateing [input_image, image] horizontally, note these two images must have
    the same height. 
    
    Note: The directory to save must exist and the data range of the image tensor must be
    in [0, 1].
    """
    image = image.cpu().numpy()
    image = np.transpose(image, [1, 2, 0])
    if do_post_process:
        image = post_process(image)
    image = np.clip(image * 255, 0, 255)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # construct image pair for comparison
    if input_image is not None:
        input_image = input_image.cpu().numpy()
        input_image = np.transpose(input_image, [1, 2, 0])
        input_image = np.clip(input_image * 255, 0, 255)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        image = cv2.hconcat([input_image, image])

    cv2.imwrite(save_path, image)


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


def convert_to_image(input: torch.Tensor, data_range: List = [0., 1.], input_order="CHW") -> np.ndarray:
    """
    Args:
        input (tensor): input tensor of shape (C, H, W).
    """
    assert isinstance(input, torch.Tensor), "Expect input as Tensor."
    
    assert input_order in ["CHW", "HWC"], "input_order must be chosen from ['CHW', 'HWC']"
    if input_order == "CHW":
        input = torch.einsum("chw->hwc", input)
    
    input = input.detach().cpu().numpy()

    input = (input - data_range[0]) / (data_range[1] - data_range[0])
    input = np.clip(input * 255, 0, 255).astype(np.uint8)
    return input



@torch.no_grad()
def test(dataloader, model, metrics: List[Dict[str, Callable]]):
    results = defaultdict(list)

    for data in tqdm.tqdm(dataloader, total=len(dataloader)):
        outputs = model(data)
        for output, record in zip(outputs, data):
            target = record["target"]
            target = convert_to_image(target)
            output = convert_to_image(output)

            for metric_name, metric_fn in metrics.items():
                results[metric_name].append(metric_fn(output, target))
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="", type=str, help="path to the config file")
    parser.add_argument('--opts', default=None, help="modify config using the command line, e.g. --opts model.name \"ResNet50\" data.batch_size=30", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = init_config(args)

    print("Configuration:")
    print(json.dumps(config, indent=4))

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = build_test_model(config["model"], device)
    resume_checkpoint(model, config["test"]["resume_checkpoint"])

    metrics = build_test_metrics(config["test"]["metrics"])

    # >>> Build Test Data Loader >>>
    cfg_test_factory = config["data_factory"]["test"]
    print(f"Testing on {cfg_test_factory['names']} datasets...")

    transforms = build_transforms(cfg_test_factory["transforms"])

    dataloader = build_test_loader(
        names=cfg_test_factory["names"],
        batch_size=cfg_test_factory["batch_size"],
        num_workers=cfg_test_factory["num_workers"],
        transforms=transforms
    )
    # <<< Build Test Data Loader <<<

    results = test(dataloader, model, metrics)


    for metric_name, records in results.items():
        print(metric_name, ": ", sum(records) / len(records))


if __name__ == '__main__':
    main()
