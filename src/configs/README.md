Details of the items used in the configuration.



"""json
{
    "trainer":{
        # save the checkpoint every 100 iterations
        "saved_period": 100,

        # configuration to monitor the performance of the model and save best
        # monitor should be chosen from ["off", "<mnt_mode> <mnt_metric>"],
        # <mnt_mode> can only be "min" or "max", 
        # <mnt_metric> is the metric key in the outputs dict returned by the model, loss by default.
        # if you want to use the metric computed on valid/test datasets, you should prepend the mode tag to it,
        # such as valid_loss, test_loss, etc.
        # if you set monitor to off or you don't specify this item,
        # the trainer will automatically save the checkpoint of the last epoch as the best model.
        "monitor": "min loss",

        "early_stop": 10,

        # path to the saved checkpoint to resume for training/testing.
        # if you do not provide this term or set it to "", training will start from scratch.
        "resume_checkpoint": "/path/to/saved_checkpoint.pt",

        # train 10 epochs
        # There is no true epoch, number of epochs will be converted to number of iterations by:
        # num_iters = epochs * iters_per_epoch
        "epochs": 10,

        # you can set the number of iterations each epoch here
        # if you set it to -1, the number will be calculated as the length of the train dataloader.
        # if the number you set is larger than the length of the train dataloader, 
        # only len(train_loader) times iterations will be excuted.
        iters_per_epoch: 1000,



    }
}
"""