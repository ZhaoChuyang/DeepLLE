Details of the items used in the configuration.



"""json
{
    "trainer":{
        # save the checkpoint every 100 iterations
        "saved_period": 100,

        # configuration to monitor the performance of the model and save best
        # monitor should be chosen from ["off", "<mnt_mode> <mnt_metric>"],
        # <mnt_mode> can only be "min" or "max", 
        # <mnt_metric> is the metric key defined in the outputs dict returned by the model, val_loss by default.
        # if you set monitor to off or you don't specify this item,
        # the trainer will automatically save the checkpoint of the last epoch as the best model.
        "monitor": "min val_loss",

        "early_stop": 10,

        # path to the saved checkpoint to resume for training/testing.
        # if you do not provide this term or set it to "", training will start from scratch.
        "resume_checkpoint": "/path/to/saved_checkpoint.pt",
    }
}
"""