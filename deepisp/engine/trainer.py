# Created on Sat Oct 08 2022 by Chuyang Zhao
import os
import copy
import glob
import time
import logging
from typing import Dict
import torch
from numpy import inf
from ..utils import TensorboardWriter, MetricTracker, check_path_exists


class BaseTrainer:
    """
    Base class for all trainers.
    """
    def __init__(self, model, optimizer, config, train_loader, valid_loader = None, lr_scheduler = None) -> None:
        self.config = config
        self.logger = logging.getLogger('train')

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self._train_loader_iter = iter(self.train_loader)

        self.valid_loader = valid_loader
        self.do_validation = False if self.valid_loader is None else True
        if self.do_validation:
            self._valid_loader_iter = iter(self.valid_loader)

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.saved_period = cfg_trainer['saved_period']
        self.save_last = cfg_trainer['save_last']
        self.eval_period = cfg_trainer['eval_period']
        self.log_period = cfg_trainer['log_period']
        self.monitor = cfg_trainer['monitor']
        self.max_eval_iters = cfg_trainer["max_eval_iters"]
        self.iters_per_epoch = cfg_trainer["iters_per_epoch"]

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
            self.mnt_metric = 'total_loss'
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

        if cfg_trainer["resume_checkpoint"]:
            self._resume_checkpoint(cfg_trainer["resume_checkpoint"])

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

    def _save_checkpoint(self, filename: str, save_best: bool = False) -> None:
        """
        Saving checkpoint, checkpoint contains:
        - iter: current iteration step
        - state_dict: state dict of the model
        - optimizer: state dict of the optimizer
        - moniter_best: currently monitered best score
        - config: config dict
        
        Args:
            filename (str): filename of the saved checkpoint.
            save_best (bool): if True, rename the saved checkpoint to 'model_best.pt'
        """
        state = {
            'iter': self.iter,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(state, path)
        self.logger.info("Saving checkpoint to: {} ...".format(path))
        
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pt')
            torch.save(state, best_path)
            self.logger.info("Saving current best to: {} ...".format(best_path))


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
        self.mnt_best = checkpoint['monitor_best']

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint['state_dict'])
        
        self.logger.info("Model's state dict loaded, missing keys: {}, unexpected keys: {}".format(missing_keys, unexpected_keys))

        try:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        except:
            self.logger.warning("Warning: Failed to read the state dict of the optimizer.")

        if self.lr_scheduler:
            self.lr_scheduler.last_epoch = self.iter - 2

        self.logger.info("Checkpoint loaded, resume training from iteration: {}".format(self.start_iter))


class Trainer(BaseTrainer):
    """
    Basic Trainer that is suitable for most of the training tasks.
    """
    def __init__(self, model, train_loader, optimizer, config, device, valid_loader=None, lr_scheduler=None):
        super().__init__(model, optimizer, config, train_loader, valid_loader, lr_scheduler)

        self.use_grad_clip = config['trainer']['use_grad_clip']
        self.ema_rate = config['trainer']['ema_rate']

        # resume or initialize EMA parameters if saving the EMA model
        if self.ema_rate:
            if config["trainer"]["resume_checkpoint"]:
                self.resume_ema_checkpoint(config["trainer"]["resume_checkpoint"])
            else:
                self.model_params = list(self.model.parameters())
                self.ema_params = copy.deepcopy(self.model_params)

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
            
        self.tb_writer.set_step(step, mode)

        for key, loss in loss_dict.items():
            metric.update(key, loss.detach().cpu().item())
        
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
            self.train_metrics.reset()
            self.valid_metrics.reset()
    
    def resume_ema_checkpoint(self, resume_path: str):
        # find ema checkpoint
        model_filename = resume_path.split("/")[-1]
        save_dir = os.path.dirname(resume_path)
        
        is_best = "best" in model_filename
        if is_best:
            ema_filename = f"ema_{self.ema_rate}_best.pt"
        else:
            step = model_filename.split(".")[0].split("_")[-1]
            ema_filename = f"ema_{self.ema_rate}_{step}.pt"
        
        ema_path = os.path.join(save_dir, ema_filename)
        if not check_path_exists(ema_path):
            raise FileNotFoundError(f"EMA checkpoint was not found in: {ema_path}")
        
        self.logger.info("Loading EMA checkpoint: {}...".format(ema_path))
        checkpoint = torch.load(ema_path)

        state_dict = checkpoint["state_dict"]

        # convert state dict to model parameters
        self.ema_params = [state_dict[name] for name, _ in self.model.named_parameters()]
        self.model_params = list(self.model.parameters())

        self.logger.info("EMA checkpoint loaded.")

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

        # evaluate model performance according to configured metric, save best checkpoint as model_best
        best = False
        if self.mnt_mode != 'off':
            
            if self.mnt_metric not in result:
                self.logger.error("Error: Metric: {} is not found in model's outputs dict: {}.".format(self.mnt_metric, result.keys()))
                raise RuntimeError("Error: Metric: {} is not found in model's outputs dict: {}.".format(self.mnt_metric, result.keys()))
            
            improved = (self.mnt_mode == 'min' and result[self.mnt_metric] <= self.mnt_best) or \
                (self.mnt_mode == 'max' and result[self.mnt_metric] >= self.mnt_best)

            if improved:
                self.mnt_best = result[self.mnt_metric]
                best = True

        filename = "model_{}.pt".format(self.iter)
        self._save_checkpoint(filename, save_best=best)
        
        if self.ema_rate:
            ema_filename = f"ema_{self.ema_rate}_{self.iter}.pt"
            self._save_ema_checkpoint(ema_filename, save_best=best)

    def _save_ema_checkpoint(self, filename: str, save_best: bool = False):
        """
        Save the state dict of the EMA model along with other training states.
        """
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = self.ema_params[i]
        
        state = {
            'iter': self.iter,
            'state_dict': state_dict,
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(state, path)
        self.logger.info("Saving checkpoint to: {} ...".format(path))
        
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, f'ema_{self.ema_rate}_best.pt')
            torch.save(state, best_path)
            self.logger.info("Saving current best to: {} ...".format(best_path))

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
        # if save ema model, keep both the last n model checkpoints and ema checkpoints.
        save_last = self.save_last * 2 if self.ema_rate else self.save_last

        checkpoints_to_clear = checkpoint_paths[:-save_last]
        for path in checkpoints_to_clear:
            os.remove(path)

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