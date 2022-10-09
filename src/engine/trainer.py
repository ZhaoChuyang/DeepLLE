# modified from https://github.com/victoresque/pytorch-template/blob/master/base/base_trainer.py
# Created on Sat Oct 08 2022 by Chuyang Zhao
import logging
import torch
from abc import abstractmethod
from numpy import inf
from ..utils import TensorboardWriter, MetricTracker, check_path_exists


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, train_loader, optimizer, config: dict) -> None:
        self.config = config
        self.logger = logging.getLogger('trainer', config['trainer']['verbosity'])

        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self._train_loader_iter = iter(self.train_loader)
    
        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.saved_period = cfg_trainer['saved_period']
        self.monitor = cfg_trainer.get('moniter', 'off')

        if cfg_trainer["iters_per_epoch"] == -1:
            try:
                self.iters_per_epoch = len(self.train_loader)
            except:
                self.logger.error("Error: The length of the data loader: {} can not be automatically determined. Please set iters_per_epoch manually in the config.".format(config["data_factory"]["train"]))
                raise RuntimeError("Error: The length of the data loader: {} can not be automatically determined. Please set iters_per_epoch manually in the config.".format(config["data_factory"]["train"]))
        else:
            try:
                if cfg_trainer["iters_per_epoch"] > len(self.train_loader):
                    self.logger.warning("Warning: iters_per_epoch: {} is larger than the length of the data loader: {}, automatically set it to the length of the data loader".format(cfg_trainer["iters_per_epoch"], len(self.train_loader)))
                    self.iters_per_epoch = len(self.train_loader)
            except:
                pass
        
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
        
        self.iter = 1
        self.start_iter = 1
        self.max_iter = self.iters_per_epoch * self.epochs + 1

        self.checkpoint_dir = cfg_trainer["ckp_dir"]

        # set tensorboard
        self.tb_writer = TensorboardWriter(cfg_trainer["log_dir"], self.logger, cfg_trainer["tensorboard"])

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        if self.config.get("resume_checkpoint", None):
            self._resume_checkpoint(self.config.get("resume_checkpoint"))

    @property
    def epoch(self):
        return self.iter // self.iters_per_epoch

    def before_epoch(self):
        """
        Execute this step when current iteration is the first iteration of the epoch,
        i.e. (self.iter - 1) % self.iters_per_epoch == 0, otherwise skip this step.
        
        """
        if (self.iter - 1) % self.iters_per_epoch == 0:
            self.train_metrics.reset()


    def after_epoch(self):
        """
        Execute this step when current iteration is the last iteration of the epoch,
        i.e. (self.iter - self.iters_per_epoch) % self.iters_per_epoch == 0, otherwise skip this step.
        """
        if (self.iter - self.iters_per_epoch) % self.iters_per_epoch == 0:
            result = self.train_metrics.result()
            valid_result = self.valid_metrics.result()
            # prepend tag 'valid' to the valid result
            valid_result = {'valid_{}'.format(key): val for key, val in valid_result.items()}
            # merge results from train and valid
            result.update(valid_result)

            log = {'epoch': self.epoch}
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
                    best = True

            if self.epoch % self.save_period == 0:
                self._save_checkpoint(save_best=best)
        
    def before_step(self):
        pass

    def after_step(self):
        pass

    def run_step(self):
        raise NotImplementedError

    def train(self):
        """
        Full train logic for calling in the main function.

        """

        for self.iter in range(self.start_iter, self.max_iter):
            self.before_epoch()
            self.before_step()
            self.run_step()
            self.after_step()
            self.after_epoch()

        self.iter += 1

    def _save_checkpoint(self, save_best=False) -> None:
        """
        Saving checkpoint, checkpoint contains:
        - iter: current iteration step
        - state_dict: state dict of the model
        - optimizer: state dict of the optimizer
        - moniter_best: currently monitered best score
        - config: config dict
        
        Args:
            save_best (bool): if True, rename the saved checkpoint to 'model_best.pth'
        """
        state = {
            'iter': self.iter,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }

        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(self.epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")


    def _resume_checkpoint(self, resume_path: str) -> None:
        """
        Resume training from saved checkpoint. Information saved in checkpoint includes:
        - iter: current iteration
        - mnt_best: best score monitered
        - state_dict: state dict of the trained model
        - optimizer: state dict of the optimizer

        Args:
            resume_path (str): checkpoint path to be resumed.
        """
        if not check_path_exists(resume_path):
            raise FileNotFoundError("Checkpoint to resume was not found in {}".format(resume_path))
                
        self.logger.info("Loading checkpoint: {}...".format(resume_path))
        checkpoint = torch.load(resume_path)

        self.iter = checkpoint['iter'] + 1
        self.start_iter = checkpoint['iter'] + 1
        self.mnt_best = checkpoint['moniter_best']

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint['state_dict'])
        
        self.logger.info("Model's state dict loaded, missing keys: {}, unexpected keys: {}".format(missing_keys, unexpected_keys))

        try:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        except:
            self.logger.warning("Warning: Failed to read the state dict of the optimizer.")

        self.logger("Checkpoint loaded, resume training from epoch: {}".format(self.start_epoch))


class Trainer(BaseTrainer):
    """
    Basic Trainer that suitable for most of the training tasks.
    """
    def __init__(self, model, metric_ftns, train_loader, optimizer, config, device, valid_loader=None, lr_scheduler=None):
        super().__init__(model, train_loader, optimizer, config)

        self.valid_loader = valid_loader
        self._valid_loader_iter = iter(self.valid_loader)
        self.do_validation = False if self.valid_loader is None else True
        self.lr_scheduler = lr_scheduler

    def next_train_iter(self):
        try:
            data = next(self._train_loader_iter)
        except StopIteration:
            self._train_loader_iter = iter(self.train_loader)
            data = next(self._train_loader_iter)
        return data

    def next_valid_iter(self):
        try:
            data = next(self._valid_loader_iter)
        except StopIteration:
            self._valid_loader_iter = iter(self.valid_loader)
            data = next(self._valid_loader_iter)
        return data

    def run_step(self):
        self.model.train()
        



