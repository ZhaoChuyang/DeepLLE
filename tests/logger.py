from deeplle.engine.launch import launch
from deeplle.data.samplers import *
from deeplle.utils import comm
import logging
from deeplle.utils.logger import setup_logger


def main():
    output = "tests/logs"
    rank = comm.get_rank()
    setup_logger(output=output, distributed_rank=rank, name="DeepLLE")
    # logger = setup_logger(output=output, distributed_rank=rank, name="DeepLLE")
    logger = logging.getLogger(__name__)
    logger.info("Hello World!")
    

if __name__ == '__main__':
    args = ()
    launch(main, 4, 1, 0, dist_url='auto', args=args)