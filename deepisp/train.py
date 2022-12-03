# Created on Sat Oct 08 2022 by Chuyang Zhao
import argparse
from distutils.command.build import build
import os
import torch
from torch import nn
import logging
import json

from .utils import init_config, setup_logger, mkdirs, get_ip_address
from .engine.trainer import Trainer
from .modeling import build_model
from .data import build_transforms, build_train_loader, build_test_loader
from .solver import build_optimizer, build_lr_scheduler


class ISPTrainer(Trainer):
    def __init__(self, config, device):
        self.device = device
        cfg_model = config["model"]
        model = self.build_model(cfg_model)

        cfg_train_factory = config["data_factory"]["train"]
        cfg_valid_factory = config["data_factory"]["valid"]
        train_loader = self.build_train_loader(cfg_train_factory)
        valid_loader = self.build_valid_loader(cfg_valid_factory)

        cfg_solver = config["solver"]
        optimizer = self.build_optimizer(cfg_solver, model)
        lr_scheduler = self.build_lr_scheduler(cfg_solver, optimizer)
        

        super().__init__(model, train_loader, optimizer, config, device, valid_loader, lr_scheduler)


    def build_model(self, cfg_model):
        # turn off testing when in training mode
        cfg_model["args"]["testing"] = False

        model = build_model(cfg_model)
        model = nn.parallel.DataParallel(model)
        model.to(self.device)
        
        return model

    def build_train_loader(self, cfg_train_factory):
        batch_size = cfg_train_factory["batch_size"]
        num_workers = cfg_train_factory["num_workers"]
        sampler = cfg_train_factory["sampler"]

        # build transforms
        cfg_transforms = cfg_train_factory["transforms"]
        transforms = build_transforms(cfg_transforms)

        # TODO: not a good practice to use transforms as the dataset argument directly.
        # considering implement a wrapper dataset which takes list of dataset, transforms, mode as argument.

        # build dataset / get dataset names
        names = cfg_train_factory["names"]

        # build dataloader
        dataloader = build_train_loader(names=names, batch_size=batch_size, num_workers=num_workers, sampler=sampler, transforms=transforms)
        return dataloader

    def build_valid_loader(self, cfg_valid_factory):
        batch_size = cfg_valid_factory["batch_size"]
        num_workers = cfg_valid_factory["num_workers"]

        # build transforms
        cfg_transforms = cfg_valid_factory["transforms"]
        transforms = build_transforms(cfg_transforms)

        # build dataset
        names = cfg_valid_factory["names"]

        # build dataloader
        dataloader = build_test_loader(names=names, batch_size=batch_size, num_workers=num_workers, transforms=transforms)
        
        return dataloader

    def build_optimizer(self, cfg_solver, model):
        name = cfg_solver["optimizer"]["name"]
        args = cfg_solver["optimizer"]["args"]
        optimizer = build_optimizer(model, name, **args)
        return optimizer

    def build_lr_scheduler(self, cfg_solver, optimizer):
        cfg_lr_scheduler = cfg_solver["lr_scheduler"]
        
        lr_scheduler = None
        if cfg_lr_scheduler:
            lr_scheduler = build_lr_scheduler(optimizer, cfg_lr_scheduler["name"], **cfg_lr_scheduler["args"])
        
        return lr_scheduler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="", type=str, help="path to the config file")
    parser.add_argument('--opts', default=None, help="modify config using the command line, e.g. --opts model.name \"ResNet50\" data.batch_size=30", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = init_config(args)

    # create the log dir and checkpoints saving dir if not exist
    mkdirs(config["trainer"]["ckp_dir"])
    mkdirs(config["trainer"]["log_dir"])

    setup_logger(config["trainer"]["log_dir"])
    logger = logging.getLogger('train')

    logger.info("Configuration:")
    logger.info(json.dumps(config, indent=4))

    if config["trainer"]["tensorboard"]:
        ip_address = get_ip_address()
        logger.info(
            f"Tensorboard is enabled, you can start tensorboard by:\n"
            f"\"tensorboard --logdir={config['trainer']['log_dir']} --port=8080 --host=0.0.0.0\".\n"
            f"You can visit http://{ip_address}:8080/ in your local broswer to watch the tensorboard.\n"
        )

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    trainer = ISPTrainer(config, device)
    trainer.train()


if __name__ == '__main__':
    main()
