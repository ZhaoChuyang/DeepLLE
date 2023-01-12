# Created on Wed Jan 04 2023 by Chuyang Zhao
import logging
import os
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel, DataParallel
from deeplle.utils import check_path_exists
from typing import Any, Dict, List, Optional


class Checkpointer:
    """
    Checkpointer manager for saving/loading checkpoints of model
    and other checkpointables like optimizer, scheduler, etc.
    """
    def __init__(
        self,
        model: nn.Module,
        save_dir: str = "",
        *,
        save_to_disk: bool = True,
        **checkpointables,
    ):
        """
        Args:
            model (nn.Module): model.
            save_dir (str): a directory to save and find checkpoints.
            save_to_disk (bool): if True, save checkpoint to disk, otherwise
                disable saving for this checkpointer. This is useful to disable
                saving on non-master processes.
            checkpointables (object): any checkpointable objects, i.e., objects
                that have the ``state_dict()`` and ``load_state_dict()`` method. For
                example, it can be used like
                `Checkpointer(model, "dir", optimizer=optimizer)`.
        """
        # save weights of bare model
        if isinstance(model, (DistributedDataParallel, DataParallel)):
            model = model.module
        self.model = model
        self.checkpointables: Dict[str, Any] = {}
        for k, v in checkpointables.items():
            self.add_checkpointable(k, v)
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        self.logger: logging.Logger = logging.getLogger(__name__)
    
    def add_checkpointable(self, key: str, checkpointable: Any) -> None:
        """
        Add checkpointable object for this checkpointer to track.

        Args:
            key (str): the key used to save the object
            checkpointable: any object with ``state_dict()`` and
                ``load_state_dict()`` method
        """
        if key in self.checkpointables:
            raise KeyError(f"Key {key} already used in the Checkpointer")
        if not hasattr(checkpointable, "state_dict"):
            raise TypeError(
                "add_checkpointable needs an object with 'state_dict()' method."
            )
        self.checkpointables[key] = checkpointable

    def save(self, name: str, save_best=False, **kwargs) -> None:
        """
        Save model and checkpointables to a file.
        """
        if not self.save_dir or not self.save_to_disk:
            return
        
        data = {}
        data["model"] = self.model.state_dict()
        for key, obj in self.checkpointables.items():
            data[key] = obj.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, name)
        with open(save_file, "wb") as f:
            torch.save(data, f)
        self.logger.info("Saving checkpoint to {}".format(save_file))

        if save_best:
            best_path = os.path.join(self.save_dir, 'model_best.pt')
            torch.save(data, best_path)
            self.logger.info("Saving current best to: {} ...".format(best_path))

    def load(self, path, checkpointables: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Load the state dict of model and checkpointables from a file.

        Args:
            path (str): path to the checkpoint file.
            checkpointables (list[str]): list of checkpointable names to load.
                If None, load all checkpointables.
        """
        if not path:
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        
        self.logger.info("Loading the model from {} ...".format(path))
        assert check_path_exists(path), "Checkpoint path does not exist: {}".format(path)
        checkpoint = self._load_file(path)
        missing_keys, unexpected_keys = self.load_state_dict(self.model, checkpoint.pop("model"))
        if missing_keys or unexpected_keys:
            self.logger.warning(f"Checkpoint loaded, missing keys: {missing_keys}, unexpected keys: {unexpected_keys}")

        for key in self.checkpointables if checkpointables is None else checkpointables:
            if key in checkpoint:
                self.logger.info(f"Loading checkpointable {key} from {path}...")
                obj = self.checkpointables[key]
                obj.load_state_dict(checkpoint.pop(key))
        
        # return any further checkpoint
        return checkpoint
    
    def _load_file(self, f: str) -> Dict[str, Any]:
        """
        Load a checkpoint file. Can be overwritten by subclasses to support
        different formats.

        Args:
            f (str): a locally mounted file path.
        Returns:
            dict: with keys "model" and optionally others that are saved by
                the checkpointer dict["model"] must be a dict which maps strings
                to torch.Tensor or numpy arrays.
        """
        return torch.load(f, map_location=torch.device("cpu"))

    @staticmethod
    def load_state_dict(model, state_dict: Dict[str, Any], prefix="") -> None:
        """
        Load model's state dict from given state_dict.

        Args:
            model (nn.Module): model to be resumed.
            state_dict (OrderedDict): state_dict to be loaded.
            prefix (str): prefix to remove from keys.
        """
        keys = sorted(state_dict.keys())
        for key in keys:
            if key.startswith(prefix):
                new_key = key[len(prefix):]
                state_dict[new_key] = state_dict.pop(key)

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        return missing_keys, unexpected_keys

    @staticmethod
    def resume_checkpoint(model, resume_path: str, ema_model: bool = False, prefix: str = "") -> None:
        """
        Resume model's checkpoint from given path.

        Args:
            model (nn.Module): model to be resumed.
            resume_path (str): path to the checkpoint.
            ema_model (bool): whether to resume the EMA model's checkpoint.
            prefix (str): prefix to remove from keys.
        """
        logger = logging.getLogger(__name__)
        assert check_path_exists(resume_path), "Resume path does not exist: {}".format(resume_path)

        logger.info("Loading {} checkpoint from {} ...".format("EMA model" if ema_model else "model", resume_path))
        checkpoint = torch.load(resume_path, map_location="cpu")
        
        if ema_model:
            assert "ema_model" in checkpoint["trainer"], "state dict of the EMA model not found in checkpoint."
            state_dict = checkpoint["trainer"]["ema_model"]
        else:
            state_dict = checkpoint["model"]
        
        missing_keys, unexpected_keys = Checkpointer.load_state_dict(model, state_dict, prefix=prefix)
        
        if missing_keys or unexpected_keys:
            logger.warning("Checkpoint loaded, missing keys: {}, unexpected keys: {}".format(missing_keys, unexpected_keys))
        else:
            logger.info("Model's state dict successfully loaded.")
