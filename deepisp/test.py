# Created on Sat Oct 08 2022 by Chuyang Zhao
import argparse
import os
import torch
from torch import nn
import logging
import json
from typing import Dict, List, Optional
import numpy as np
import cv2
import tqdm

from .utils import init_config, setup_logger, mkdirs, check_path_exists, check_path_is_image
from .modeling import build_model
from .data import build_transforms, build_test_loader, CommISPDataset


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


def read_images(img_dir):
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
                

# def process_dataset(dataset_dicts: List[Dict], transforms):
#     """
#     1. read image
#     2. apply given transforms on images
#     """
#     for record in dataset_dicts:
#         image = Image.open(record["image_path"])
#         image = transforms(image)
#         record["image"] = image
    
#     return dataset_dicts


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="", type=str, help="path to the config file")
    parser.add_argument('--opts', default=None, help="modify config using the command line, e.g. --opts model.name \"ResNet50\" data.batch_size=30", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = init_config(args)

    # create the test output directory if it does not exist.
    mkdirs(config["test"]["save_dir"])

    # setup_logger(config["trainer"]["log_dir"])
    # logger = logging.getLogger('train')

    print("Configuration:")
    print(json.dumps(config, indent=4))

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = build_test_model(config["model"], device)
    resume_checkpoint(model, config["test"]["resume_checkpoint"])

    # >>> Build Test Data Loader >>>
    cfg_test_factory = config["data_factory"]["test"]

    transforms = build_transforms(cfg_test_factory["transforms"])

    # dataset_dicts = read_images(cfg_test_factory["img_dir"])
    dataset_dicts = load_data(cfg_test_factory["img_dir"], config["test"]["save_dir"])

    dataset = CommISPDataset(dataset_dicts, False, transforms)
    dataloader = build_test_loader(dataset=dataset, batch_size=cfg_test_factory["batch_size"], num_workers=0)
    # <<< Build Test Data Loader <<<

    with torch.no_grad():
        for data in tqdm.tqdm(dataloader, total=len(dataloader)):
            outputs = model(data)

            for output, record in zip(outputs, data):
                save_path = record["save_path"]
                input_image = None
                if config["test"]["save_pair"]:
                    input_image = record["image"]
                save_image(save_path, output, input_image=input_image)


if __name__ == '__main__':
    main()
