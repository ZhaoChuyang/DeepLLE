# Created on Sat Oct 08 2022 by Chuyang Zhao
import os
import json
import logging
import logging.config
from .. import check_path_exists, read_json, mkdirs


__all__ = ["setup_logger"]


def setup_logger(save_dir, log_config="utils/logger/logger_config.json", default_level=logging.INFO):
    """Setup logger with configuration

    Create two loggers, one for stdout logging, one for file logging.

    """
    if check_path_exists(log_config):
        config = read_json(log_config)

        for handler in config["handlers"].values():
            if 'filename' in handler:
                handler['filename'] = os.path.join(save_dir, handler['filename'])

        logging.config.dictConfig(config)
    else:
        print("Warning: log config does not exist in {}.".format(log_config))
        logging.basicConfig(level=default_level)
