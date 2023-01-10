# Created on Sat Oct 08 2022 by Chuyang Zhao
import os
import copy
import glob
import time
import logging
from typing import Dict
import weakref
import torch
from torch import nn
import numpy as np
from numpy import inf
from deeplle.utils import TensorboardWriter, MetricTracker, comm
from deeplle.utils.nn_utils import get_model_info
from deeplle.utils.checkpoint import Checkpointer
from torch.nn.parallel import DistributedDataParallel, DataParallel


class BaseTrainer:
    """
    Base class for all trainers.
    """
    def __init__(self, model, optimizer, config, train_loader, valid_loader = None, lr_scheduler = None) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)

        cfg_trainer = config['trainer']
        self.epochs: int = int(cfg_trainer['epochs'])
        self.saved_period: int = int(cfg_trainer['saved_period'])
        self.save_last: int = int(cfg_trainer['save_last'])
        self.eval_period: int = int(cfg_trainer['eval_period'])
        self.log_period: int = int(cfg_trainer['log_period'])
        self.monitor: str = cfg_trainer['monitor']
        self.max_eval_iters: int = int(cfg_trainer["max_eval_iters"])
        self.iters_per_epoch: int = int(cfg_trainer["iters_per_epoch"])

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self._train_loader_iter = iter(self.train_loader)

        self.valid_loader = valid_loader
        # do not evaluate on validation set if valid loader is not provided or eval_period <= 0
        self.do_validation = False if self.valid_loader is None or self.eval_period <= 0 else True
        if self.do_validation:
            self._valid_loader_iter = iter(self.valid_loader)

        # Try infer the number of iterations per epoch from the dataset.
        # If the length of the dataset cannot be determined, which happens
        # when the dataset is iterable-style, raise an runtime error.
        if self.iters_per_epoch == -1:
            try:
                self.iters_per_epoch = len(self.train_loader)
            except:
                raise RuntimeError("Error: The length of the data loader: {} can not be automatically determined. Please set iters_per_epoch manually in the config.".format(config["data_factory"]["train"]))
        
        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_metric = 'last'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']
            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        
        self.iter = 1
        self.start_iter = 1
        self.max_iter = int(self.iters_per_epoch * self.epochs) + 1

        self.checkpoint_dir = cfg_trainer["ckp_dir"]

        # set tensorboard
        self.tb_writer = TensorboardWriter(cfg_trainer["log_dir"], self.logger, cfg_trainer["tensorboard"])

        self.train_metrics = MetricTracker(writer=self.tb_writer)
        self.valid_metrics = MetricTracker(writer=self.tb_writer)

    @property
    def epoch(self):
        return int(self.iter // self.iters_per_epoch)

    def before_epoch(self):
        """
        Execute this procedure when current iteration is the first iteration of the epoch,
        i.e. (self.iter - 1) % self.iters_per_epoch == 0, otherwise skip this procedure.
        """
        if (self.iter - 1) % self.iters_per_epoch == 0:
            pass

    def after_epoch(self):
        """
        Execute this procedure when current iteration is the last iteration of the epoch,
        i.e. (self.iter - self.iters_per_epoch) % self.iters_per_epoch == 0, otherwise skip this procedure.
        """
        if (self.iter - self.iters_per_epoch) % self.iters_per_epoch == 0:
            pass
    
    def before_step(self):
        self.model.train()

    def after_step(self):
        pass

    def run_step(self):
        raise NotImplementedError

    def train(self):
        """
        Full train logic for calling in the entrance function.

        By dafault our training process is iteration-based,
        currently `self.before_epoch()` and `self.after_epoch()`
        do nothing, but provide an interface for the compatibility
        of epoch-based training. If you want to train the model
        in epoch-based way, you can override these two methods.
        """

        for self.iter in range(self.start_iter, self.max_iter):
            self.before_epoch()
            self.before_step()
            self.run_step()
            self.after_step()
            self.after_epoch()

        # increment the iter counter to mark the comleting of training
        self.iter += 1

    def state_dict(self):
        ret = {
            'iteration': self.iter,
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        if self.lr_scheduler:
            ret['lr_scheduler'] = self.lr_scheduler.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        self.iter = state_dict['iteration'] + 1
        self.start_iter = self.iter
        self.mnt_best = state_dict['monitor_best']
        self.optimizer.load_state_dict(state_dict['optimizer'])
        if self.lr_scheduler:
            self.lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
        
        self.logger.info("Resume training from iteration {}".format(self.iter))


class SimpleTrainer(BaseTrainer):
    """
    Simple Trainer that is suitable for most of the training tasks.
    """
    def __init__(self, model, train_loader, optimizer, config, valid_loader=None, lr_scheduler=None):
        super().__init__(model, optimizer, config, train_loader, valid_loader, lr_scheduler)

        self.print_network()

        self.use_grad_clip = config['trainer']['use_grad_clip']
        self.ema_rate = config['trainer']['ema_rate']

        is_main_process = comm.is_main_process()
        self.checkpointer = Checkpointer(
            self.model, 
            save_dir=self.checkpoint_dir, 
            save_to_disk=is_main_process,
            trainer=weakref.proxy(self),
        )

        if self.ema_rate:
            # unwrap the model from DataParallel or DistributedDataParallel
            model = self.get_bare_model()
            self.model_params = list(model.parameters())
            # initialize the EMA model
            self.ema_model = copy.deepcopy(model)
            self.ema_params = list(self.ema_model.parameters())

        if config["trainer"]["resume_checkpoint"]:
            self.checkpointer.load(config["trainer"]["resume_checkpoint"])

    @comm.master_only
    def print_network(self):
        model_info, num_params = get_model_info(self.model)
        self.logger.info(f"Model Information:\n{model_info}")
        self.logger.info(f"Number of model's parameters: {num_params}")

    def get_bare_model(self) -> nn.Module:
        model = self.model
        if isinstance(self.model, (DataParallel, DistributedDataParallel)):
            model = model.module  # type: ignore
        return model

    def _update_ema(self, target_params, source_params, rate=0.99):
        for targ, src in zip(target_params, source_params):
            targ.detach().mul_(rate).add_(src, alpha=1 - rate)

    def forward(self, data):
        return self.model(data)

    def run_step(self):
        assert self.model.training, "[Trainer] model was changed to eval mode!"

        data = next(self._train_loader_iter)

        """
        If you want to do something with the losses or compute metrics based on
        model's outputs, you can wrap the model.
        """
        loss_dict, output_dict = self.forward(data)

        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": losses}
        else:
            losses = sum(loss_dict.values())
            loss_dict["total_loss"] = losses
        
        if isinstance(output_dict, torch.Tensor):
            outputs = output_dict
            output_dict = {"outputs": outputs}

        self.optimizer.zero_grad()
        losses.backward()

        # TODO: wrap the optimizer to do gradient clip
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.01)

        self.optimizer.step()

        if self.ema_rate:
            self._update_ema(self.ema_params, self.model_params, self.ema_rate)

        self.write_metrics('train', self.iter, loss_dict, output_dict)

    def write_metrics(self, mode, step, loss_dict: Dict, output_dict: Dict):
        assert mode in ['train', 'valid'], "Mode can only be 'train' or 'valid'. Got {}.".format(mode)
        
        if mode == 'train':
            metric = self.train_metrics
        else:
            metric = self.valid_metrics

        loss_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
        output_dict = {k: v.detach().cpu().item() for k, v in output_dict.items()}

        all_loss_dict = comm.gather(loss_dict)
        all_output_dict = comm.gather(output_dict)

        if comm.is_main_process():
            # average the losses across all processes
            loss_dict = {
                k: np.mean([x[k] for x in all_loss_dict]) for k in all_loss_dict[0].keys()
            }
            # average the outputs across all processes
            output_dict = {
                k: np.mean([x[k] for x in all_output_dict]) for k in all_output_dict[0].keys()
            }

            total_losses_reduced = sum(loss_dict.values())
            if not np.isfinite(total_losses_reduced):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration={self.iter}!\n"
                    f"loss_dict = {loss_dict}"
                )
            
            self.tb_writer.set_step(step, mode)

            for key, loss in loss_dict.items():
                metric.update(key, loss)
        
            for key, output in output_dict.items():
                metric.update(key, output)

    def after_step(self):
        super().after_step()

        # update the learning rate on iteration boundaries.
        if self.lr_scheduler:
            self.lr_scheduler.step()

        # print training information to the screen periodically.
        if self.iter % self.log_period == 0:
            self.logger.info('Epoch: {}, Train Iteration: [{}/{} ({:.0f}%)], Loss: {:.6f}'.format(
                    self.epoch,
                    self.iter,
                    self.max_iter,
                    100 * self.iter / self.max_iter,
                    self.train_metrics.avg('total_loss')))

        # evaluate on the validation dataset periodically if validation dataset is provided.
        if self.do_validation and self.iter % self.eval_period == 0:
            self.do_eval()

        # save the checkpoint periodically.
        if self.iter % self.saved_period == 0:
            self.save_checkpoint()
            if self.save_last != -1:
                self.clear_checkpoints()

        # after saving the checkpoint, reset the metric tracker.
        if self.iter % self.log_period == 0:
            if comm.is_main_process():
                self.train_metrics.reset()
                self.valid_metrics.reset()

    def save_checkpoint(self):
        """
        Save the current model checkpoint to "model_{iter_num}.pt" and the best model
        checkpoint to "model_best.pt".
        If saving EMA model, save the EMA checkpoint to "ema_{ema_rate}_{iter_num}.pt" and 
        the best checkpoint to "ema_{ema_rate}_best.pt".
        """
        result = self.train_metrics.result()
        if self.do_validation:
            valid_result = self.valid_metrics.result()
            valid_result = {'valid_{}'.format(key): val for key, val in valid_result.items()}
            # merge valid and train results
            result.update(valid_result)

        # print results to the screen and save the results into log file
        result['iter'] = self.iter
        result['epoch'] = self.epoch
        for key, value in result.items():
            self.logger.info('{:15s}: {}'.format(str(key), value))

        # evaluate model performance according to monitor config, save best checkpoint as `model_best.pt`
        # if monitor is disabled, always save the last checkpoint as the best checkpoint.
        if self.mnt_mode == 'off':
            best = True
        else:
            best = False
            if self.mnt_metric not in result:
                self.logger.error("Error: Metric: {} is not found in model's outputs dict: {}.".format(self.mnt_metric, result.keys()))
                raise RuntimeError("Error: Metric: {} is not found in model's outputs dict: {}.".format(self.mnt_metric, result.keys()))
            
            improved = (self.mnt_mode == 'min' and result[self.mnt_metric] <= self.mnt_best) or \
                (self.mnt_mode == 'max' and result[self.mnt_metric] >= self.mnt_best)

            if improved:
                self.mnt_best = result[self.mnt_metric]
                best = True
        
        # only save to disk in the main process
        filename = "model_{}.pt".format(self.iter)
        self.checkpointer.save(filename, save_best=best)
    
    @comm.master_only
    def clear_checkpoints(self):
        """
        Remove earlier checkpoints and keep exactly `self.save_last` checkpoints
        if `self.save_last` is not set to -1.
        """
        checkpoint_paths = []
        for path in glob.glob(f"{self.checkpoint_dir}/*.pt"):
            if "best" in path.split("/")[-1]: 
                continue
            checkpoint_paths.append(path)
        
        checkpoint_paths = sorted(checkpoint_paths, key=lambda x: os.path.getmtime(x))
        
        assert self.save_last > 0 and isinstance(self.save_last, int)
        checkpoints_to_clear = checkpoint_paths[:-self.save_last]
        
        for path in checkpoints_to_clear:
            os.remove(path)

    def state_dict(self):
        ret = super().state_dict()
        if self.ema_rate:
            ret["ema_model"] = self.ema_model.state_dict()
        return ret
    
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if self.ema_rate:
            self.logger.info("Loading EMA model from checkpoint...")
            missing_keys, unexpected_keys = Checkpointer.load_state_dict(self.ema_model, state_dict["ema_model"])
            assert missing_keys == [] and unexpected_keys == [], "Error: EMA model's state dict is not compatible with the checkpoint."
    
    @torch.no_grad()
    def do_eval(self):
        self.model.eval()

        # inference dataset must have a fixed length
        if self.max_eval_iters != -1:
            total = min(self.max_eval_iters, len(self.valid_loader))
        else:
            total = len(self.valid_loader)
        
        start_time = time.perf_counter()

        for idx, data in enumerate(self.valid_loader):
            loss_dict, output_dict = self.forward(data)

            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": losses}
            else:
                losses = sum(loss_dict.values())
                loss_dict.update({"total_loss": losses})
            
            if isinstance(output_dict, torch.Tensor):
                outputs = output_dict
                output_dict = {"outputs": outputs}

            self.write_metrics('valid', (self.iter // self.eval_period - 1) * total + idx, output_dict, loss_dict)

            # early stop
            if idx == self.max_eval_iters:
                break
        
        total_time = time.perf_counter() - start_time

        self.logger.info(
            "Evaluation completed. Total inference time: {}({:.6f} s/iter)".format(
                total_time, total_time / total
                )
            )

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        """
        Returns:
            DatasetEvaluator or None

        It is not implemented by default.
        """
        raise NotImplementedError("Please implement `build_evaluator()` in subclass if you want to do test by the trainer.")
