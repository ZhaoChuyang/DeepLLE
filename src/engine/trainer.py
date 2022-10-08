# modified from https://github.com/victoresque/pytorch-template/blob/master/base/base_trainer.py
# Created on Sat Oct 08 2022 by Chuyang Zhao
import logging
import torch
from abc import abstractmethod
from numpy import inf
from ..utils import TensorboardWriter, check_path_exists


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, dataloader, optimizer, config: dict) -> None:
        self.config = config
        self.logger = logging.getLogger('trainer', config['trainer']['verbosity'])

        self.model = model
        self.optimizer = optimizer
    
        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.saved_period = cfg_trainer['saved_period']
        self.monitor = cfg_trainer.get('moniter', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)

            if self.early_stop <= 0:
                self.early_stop = inf
        
        self.start_epoch = 1

        self.checkpoint_dir = cfg_trainer["ckp_dir"]

        # set tensorboard
        self.tb_writer = TensorboardWriter(cfg_trainer["log_dir"], self.logger, cfg_trainer["tensorboard"])

        if self.config.get("resume_checkpoint", None):
            self._resume_checkpoint(self.config.get("resume_checkpoint"))
    
    @abstractmethod
    def _train_epoch(epoch):
        """
        Training logic for one epoch, each Trainer for use must implement this method.

        Args:
            epoch (int): Current epoch number.
        """
        raise NotImplementedError

    def train(self):
        """
        Full train logic for calling in the main function.

        """
        not_improved_count = 0

        for epoch in range(self.start_epoch, self.epochs+1):
            result = self._train_epoch(epoch)

            log = {'epoch': epoch}
            log.update(result)
            
            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':

                if self.mnt_metric not in log["outputs"]:
                    self.logger.error("Error: Metric: {} is not found in model's outputs dict: {}.".format(self.mnt_metric, log["outputs"].keys()))
                    raise RuntimeError("Error: Metric: {} is not found in model's outputs dict: {}.".format(self.mnt_metric, log["outputs"].keys()))
                    
                improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                
                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break
            
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
            
    

    def _resume_checkpoint(self, resume_path: str) -> None:
        """
        Resume training from saved checkpoints

        Args:
            resume_path (str): checkpoint path to be resumed.
        """
        if not check_path_exists(resume_path):
            raise FileNotFoundError("Checkpoint to resume was not found in {}".format(resume_path))
                
        self.logger.info("Loading checkpoint: {}...".format(resume_path))
        checkpoint = torch.load(resume_path)

        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['moniter_best']

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint['state_dict'])
        
        self.logger.info("Model's state dict loaded, missing keys: {}, unexpected keys: {}".format(missing_keys, unexpected_keys))

        try:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        except:
            self.logger.warning("Warning: Failed to read the state dict of the optimizer.")

        self.logger("Checkpoint loaded, resume training from epoch: {}".format(self.start_epoch))

