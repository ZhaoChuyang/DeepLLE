#!/usr/bin/env python
# Created on Thu Jan 06 2023 by Chuyang Zhao
import os
import sys
import json
import argparse
from deeplle.engine.launch import launch
from deeplle.engine.trainer import SimpleTrainer
from deeplle.utils.logger import setup_logger
from deeplle.utils import comm
from deeplle.utils import mkdirs, get_ip_address, init_config
from deeplle.utils.nn_utils import get_model_info
from deeplle.modeling import build_model, create_ddp_model
from deeplle.solver import build_optimizer, build_lr_scheduler
from deeplle.data import build_image_transforms, build_isp_train_loader, build_test_loader


def default_argument_parser(epilog=None):
    """
    Create a parser with some common arguments used by detectron2 users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
Examples:

Run on single machine:
    $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml

Change some config options:
    $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001

Run on multiple machines:
    (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
    (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
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


class DefaultISPTrainer(SimpleTrainer):
    def __init__(self, config):
        cfg_model = config["model"]
        model = self.build_model(cfg_model)

        cfg_train_factory = config["data_factory"]["train"]
        cfg_valid_factory = config["data_factory"]["valid"]
        dataset_type = config["data_factory"]["type"]
        train_loader = self.build_train_loader(cfg_train_factory, dataset_type)
        valid_loader = self.build_valid_loader(cfg_valid_factory, dataset_type)

        cfg_solver = config["solver"]
        optimizer = self.build_optimizer(cfg_solver, model)
        lr_scheduler = self.build_lr_scheduler(cfg_solver, optimizer)
        
        super().__init__(model, train_loader, optimizer, config, valid_loader, lr_scheduler)

    @classmethod
    def build_model(cls, cfg_model):
        # turn off testing when in training mode
        cfg_model["args"]["testing"] = False

        model = build_model(cfg_model)
        model = create_ddp_model(model, broadcast_buffers=False)
        
        return model

    @classmethod
    def build_train_loader(cls, cfg_train_factory, type):
        dataloader = build_isp_train_loader(cfg_train_factory, type=type)
        return dataloader

    @classmethod
    def build_valid_loader(cls, cfg_valid_factory, type):
        batch_size = cfg_valid_factory["batch_size"]
        num_workers = cfg_valid_factory["num_workers"]

        # build transforms
        cfg_transforms = cfg_valid_factory["transforms"]
        transforms = build_image_transforms(cfg_transforms)

        # build dataset
        names = cfg_valid_factory["names"]

        # build dataloader
        dataloader = build_test_loader(names=names, batch_size=batch_size, num_workers=num_workers, transforms=transforms)
        
        return dataloader

    @classmethod
    def build_optimizer(cls, cfg_solver, model):
        name = cfg_solver["optimizer"]["name"]
        args = cfg_solver["optimizer"]["args"]
        optimizer = build_optimizer(model, name, **args)
        return optimizer

    @classmethod
    def build_lr_scheduler(cls, cfg_solver, optimizer):
        cfg_lr_scheduler = cfg_solver["lr_scheduler"]
        
        lr_scheduler = None
        if cfg_lr_scheduler:
            lr_scheduler = build_lr_scheduler(optimizer, cfg_lr_scheduler["name"], **cfg_lr_scheduler["args"])
        
        return lr_scheduler


def default_setup(config):
    # create the log dir and checkpoints saving dir if not exist
    mkdirs(config["trainer"]["ckp_dir"])
    mkdirs(config["trainer"]["log_dir"])

    rank = comm.get_rank()
    logger = setup_logger(config["trainer"]["log_dir"], rank)

    logger.info("Configuration:")
    logger.info(json.dumps(config, indent=4))

    if config["trainer"]["tensorboard"]:
        ip_address = get_ip_address()
        logger.info(
            f"Tensorboard is enabled, you can start tensorboard by:\n"
            f"$ tensorboard --logdir={config['trainer']['log_dir']} --port=8080 --host=0.0.0.0\n"
            f"You can visit http://{ip_address}:8080/ in your local broswer to watch the tensorboard.\n"
        )


def main(args):
    config = init_config(args)
    default_setup(config)
    
    trainer = DefaultISPTrainer(config)
    trainer.train()


if __name__ == "__main__":
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
