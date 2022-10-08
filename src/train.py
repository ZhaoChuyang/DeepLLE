# Created on Sat Oct 08 2022 by Chuyang Zhao
import argparse
import os
from utils import init_config, setup_logger, mkdirs




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="", type=str, help="path to the config file")
    parser.add_argument('--opts', default=None, help="modify config using the command line, e.g. --opts model.name \"ResNet50\" data.batch_size=30", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = init_config(args)

    # create the log dir and checkpoints saved dir if not exist
    mkdirs(config["trainer"]["ckp_dir"])
    mkdirs(config["trainer"]["log_dir"])

    setup_logger(config["trainer"]["log_dir"])
    
    print(config)

if __name__ == '__main__':
    main()
