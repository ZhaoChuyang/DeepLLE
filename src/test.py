# Created on Sat Oct 08 2022 by Chuyang Zhao
import argparse
from distutils.command.build import build
import os
import torch
from torch import nn
import logging
import json
from typing import Dict, List
import numpy as np
import cv2

from .utils import init_config, setup_logger, mkdirs, check_path_exists, check_path_is_image
from .engine.trainer import Trainer
from .modeling import build_model
from .data import build_transforms, build_train_loader, build_test_loader, CommISPDataset



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

def save_image(save_path: str, image: torch.Tensor):
    """
    """
    image = image.cpu().numpy()
    image = np.transpose(image, [1, 2, 0])
    image = np.clip(image * 255, 0, 255)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="", type=str, help="path to the config file")
    parser.add_argument('--opts', default=None, help="modify config using the command line, e.g. --opts model.name \"ResNet50\" data.batch_size=30", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = init_config(args)

    # create the log dir and checkpoints saved dir if not exist
    mkdirs(config["trainer"]["ckp_dir"])
    mkdirs(config["trainer"]["log_dir"])
    mkdirs(config["test"]["save_dir"])

    setup_logger(config["trainer"]["log_dir"])
    logger = logging.getLogger('train')

    logger.info("Configuration:")
    logger.info(json.dumps(config, indent=4))

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = build_test_model(config["model"], device)
    resume_checkpoint(model, config["test"]["resume_checkpoint"])

    # >>> Build Test Data Loader >>>
    cfg_test_factory = config["data_factory"]["test"]

    transforms = build_transforms(cfg_test_factory["transforms"])

    dataset_dicts = read_images(cfg_test_factory["img_dir"])
    dataset = CommISPDataset(dataset_dicts, False, transforms)
    dataloader = build_test_loader(dataset=dataset, batch_size=cfg_test_factory["batch_size"], num_workers=0)
    # <<< Build Test Data Loader <<<

    with torch.no_grad():
        for data in dataloader:
            outputs = model(data)
            filenames = [r["image_path"].split("/")[-1] for r in data]
            
            for filename, output in zip(filenames, outputs):
                save_path = os.path.join(config["test"]["save_dir"], filename)
                save_image(save_path, output)


if __name__ == '__main__':
    main()
