Details of the configuration options.

```python
{
    # Experiment name. All checkpoints and logs of this experiment will be saved in a folder
    # with the experiment name. Change this option every time you want to run a different experiment.
    "name": "llie_base",
    # Base config. All options in base config will be merged with the current config. Recursive config merging is supported.
    "base": "base.json",

    #------------------------------------------------------
    # The following are the dataset and dataloader settings
    #------------------------------------------------------
    "data_factory": {
        # Training dataset settings
        "train": {
            # Dataset names, can be both a dataset name string for a single dataset, or a list of dataset
            # names for multiple datasets, e.g. "names": "lol_train" or "names": ["lol_train", "mbllen_noisy"].
            "names": [
                "lol_train", "mbllen_noisy", "ve_lol_real_train", "sice_all", "fivek_all"
            ],
            # Batch size on a single GPU. In multi-gpu training setting, the total batch size equals `batch_size * num_gpus`.
            "batch_size": 32,
            # Number of workers of the data loader for each GPU. 
            "num_workers": 4,
            # Data sampler. Currently only supports `TrainingSampler` and `BalancedSampler`. Differece between these two samplers can be refered to `deeplle/data/samplers/distributed_sampler.py`.
            "sampler": "TrainingSampler",
            # Transforms applied on the images. All availabe transforms can be found in `deeplle/data/transforms/transforms`.
            "transforms": [
                {
                    "name": "RandomCrop",
                    "args": {
                        "size": 256
                    }
                },
                {
                    "name": "ToTensor",
                    "args": {}
                },
            ]
        },
        # Validation dataset settings, options are same with the training dataset
        "valid": {
            # Dataset names
            "names": "lol_val",
            # Batch size
            "batch_size": 4,
            # Number of workers
            "num_workers": 0,
            # Transforms applied on validation images
            "transforms": [
                {
                    "name": "ToTensor",
                    "args": {}
                }
            ]
        },
        # Testing dataset settings, options are same with the training dataset
        "test": {
            # Dataset names
            "names": "lol_val",
            # Batch size
            "batch_size": 1,
            # Number of workers
            "num_workers": 0,
            # Transforms applied on validation images
            "transforms": [
                {
                    "name": "ToTensor",
                    "args": {}
                }
            ]
        }
    },
    
    #-------------------------------------------------
    # The following are the network structure settings
    #-------------------------------------------------
    "model": {
        # Model name. Models must be registered in `MODEL_REGISTRY` before used in config.
        "name": "UNetBaseline",
        # Arguments for creating the model
        "args": {
            "bilinear": true,
            "depth": 2,
            "base_dim": 16
        }
    },

    #---------------------------------------
    # The following are the trainer settings
    #---------------------------------------
    "trainer": {
        # Number of training epochs. Because the training of DeepLLE is iteration-based,
        # the number of epochs will be converted to the number of iterations in DeepLLE
        # by `num_of_iterations = epochs * iters_per_epoch`. Best practice is to set the
        # number of epochs to 1 and set the total training iterations in `iters_per_epoch`.
        "epochs": 100,
        # EMA model updating rate. Set to null to disable updating EMA model.
        "ema_rate": 0.99,
        # The number of iterations in one epoch. If you set it to -1, it will be automatically
        # set to the length of the training dataset. If the length of the dataset can not
        # be automatically inferred, e.g. the dataset is iteration-based, exception will be raised.
        "iters_per_epoch": -1,
        # Checkpoint save period. Because the training is iteration-based, the checkpoint will
        # be saved every `save_period` iterations.
        "saved_period": 500,
        # The last n checkpoint files to keep, this does not include the best checkpoint. If you
        # want to keep all checkpoints, set `save_last` option to -1.
        "save_last": 3,
        # Evaluation period. Do evaluation on the validation set every `eval_peiord` iterations.
        # You can disable evaluating on validation set by setting `eval_period` to -1.
        "eval_period": 100,
        # Log period. Log the training records every `log_period` iterations.
        "log_period": 50,
        # Directory for saving training checkpoints, logs and testing files. It is relative to
        # the directory of this config file.
        "save_dir": "../../event_logs",
        # Monitor config for tracking the performance of the model. The best model checkpoint is saved
        # accourding to the monitored results. `monitor` should be a string in format `<mnt_mode> <mnt_metric>`,
        # where <mnt_mode> should be one of ["min", "max"], determining whether the best performance equals to
        # the max or min value of the monitered metric. You can disable monitoring by setting `monitor` to `off`,
        # in which way the lastest checkpoint will always be saved as the best checkpoint.
        "monitor": "min total_loss",
        # Maximum evaluation iterations. When do evaluation on the validation set, the maximum iterations to run.
        # This is useful when the validation set is extremely large and you don't want to evaluate on the full set
        # for each validation stage. You can set `max_eval_iters` to -1 to evaluate on the full validation set.
        "max_eval_iters": -1,
        # Whether record the training logs to the tensorboard. You must have tensorboard installed beforehand.
        "tensorboard": true,
        # The checkpoint to resume training from. It can be an absolute path or a relative path located in the
        # `save_dir` directory. If you want to training from scratch, you can set this to empty string or null.
        "resume_checkpoint": "model_best.pt",
        # Whether use gradient clip when computing the gradient. This may help to avoid gradient exploding problem.
        "use_grad_clip": false
    },

    #--------------------------------------
    # The following are the solver settings
    #--------------------------------------
    "solver": {
        # The optimizer related settings, you can refer to torch.nn.optim for details about all
        # available optimizers and their arguments.
        "optimizer": {
            # Optimizer class name
            "name": "Adam",
            # Arguments
            "args": {
                "lr": 3e-3
            }
        },
        # LR scheduler related settings. Currently we only provide "WarmupMultiStepLR" and "WarmupCosineLR"
        # two schedulers. They are overrided to support warm up. You can find more details in `solver/lr_scheduler.py`.
        # If you do not want to use LR scheduler, you can set `lr_scheduler` to null. 
        "lr_scheduler": {
            # Scheduler class name. Availabe chooses: ["WarmupMultiStepLR", "WarmupCosineLR"].
            "name": "WarmupMultiStepLR",
            # Scheduler arguments, refer to `solver/lr_scheduler.py` for more details.
            "args": {
                "milestones": [
                    5000,
                    30000,
                    100000
                ],
                "warmup_iters": 0,
                "warmup_factor": 0.1
            }
        }
    },
    #------------------------------------------------------------------------------------------------
    # The following are the test settings. Testing will run on the test set defined in `data_factory`
    #------------------------------------------------------------------------------------------------
    "test": {
        # The checkpoint to resume training from. It can be an absolute path or a relative path located in the
        # `save_dir` directory. If you want to training from scratch, you can set this to empty string or null.
        "resume_checkpoint": "model_best.pt",
        # Whether to load the EMA model. Note the EMA model must be saved in the checkpoint, typically with key
        # checkpoint["trainier"]["ema_model"].
        "resume_ema_model": false,
        # Metrics computed on the test dataset. Currently supported metrics are "MAE", "SSIM", "PSNR". "metrics"
        # can be either a list of the metric names, e.g. ["MAE", "SSIM], in which case metric function with default
        # arguments will be called, or a list of metric dicts, each dict should contain keys "name" and "args", 
        # where "name" is the metric class name and "args" is the keyword arguments dict of certain metric. You can
        # find more details about the arguments in `modeling/metrics`.
        "metrics": [
            {
                "name": "MAE",
                "args": {}
            },
            {
                "name": "SSIM",
                "args": {
                    "crop_border": 0,
                    "input_order": "HWC"
                }
            },
            {
                "name": "PSNR",
                "args": {}
            }
        ]
    },
    #------------------------------------------------------------------------------------------------------
    # The following are inference settings. Inference is conducted on the `infer.data`. Inference data does
    # not contain any ground-truth, so no metric or loss will be computed, instead results will be saved.
    #-------------------------------------------------------------------------------------------------------
    "infer": {
        # Inference data
        "data": {
            # Directory containing images for inference. All images found recursively in the `img_dir` will be inferenced.
            "img_dir": "/home/chuyang/Workspace/datasets/test",
            # Batch size when infering.
            "batch_size": 1,
            # Number of the dataloader workers.
            "num_workers": 0,
            # Transforms applied to the inference data.
            "transforms": [
                {
                    "name": "ToTensor",
                    "args": {}
                }
            ]
        },
        # The checkpoint to resume training from. It can be an absolute path or a relative path located in the
        # `save_dir` directory. If you want to training from scratch, you can set this to empty string or null.
        "resume_checkpoint": "model_best.pt",
        # Whether to load the EMA model. Note the EMA model must be saved in the checkpoint, typically with key
        # checkpoint["trainier"]["ema_model"].
        "resume_ema_model": false,
        # Directory to save the infered results. It should be a relative path located in "{trainer.save_dir}/test"
        # or a absolute path. Leaving it empty string will save all results in "{trainer.save_dir}/test".
        "save_dir": "",
        # Whether save input image along with output image for comparison.
        "save_pair": true
    }
}
```
