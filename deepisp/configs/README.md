Details of the configuration options.


```python
{
    "trainer":{
        """
        Period to save the checkpoint.

        Note: our training is iteration-based, so the following setting
        will save our checkpoint every 100 iterations.
        """
        "saved_period": 100,


        """
        Configuration to monitor the performance of the model and save the best checkpoint as "model_best.pt".

        "moniter" should be a string, available choices include "off" and "<mnt_mode> <mnt_metric>".
            * "off" means turn the monitor off, which will result the trainer not to save the best
              model as "model_best.pt".
            * "<mnt_mode> <mnt_metric>" is the indicator telling the moniter to moniter the changes
              of recorded metric: <mnt_metric>, and save the best model based on this metric.
              <mnt_mode> is the way to measure <mnt_metric>, it has two valid options "min" and "max".
              <mnt_metric> is the metric type which will be used to measure the performance of the model.
              
              e.g. "min loss" means measuring the performace of the model based on the its current loss,
              and the a more minimum loss means a better model.

              Note: the corresponding metric on valid dataset is prepended with "valid_", for example the
              loss computed on valid dataset is called "valid_loss". 
        """
        "monitor": "min loss",

        """
        Path to the saved checkpoint to load for resuming training. Delete this term
        or set it to "" to train from scratch.
        """
        "resume_checkpoint": "/path/to/saved_checkpoint.pt",

        """
        Number of training epochs

        There is no epoch concept in our trainer, the number of epochs will be converted to
        the number of total traing iterations by:
            total_iters = epochs * iters_per_epoch

        Note: Iteration-based training is the suggested way, in which
        you can set "epochs" to 1, and set the iters_per_epoch as the
        total training iterations you need.
        """
        "epochs": 10,

        """
        Training iterations per epoch.

        You can set it to -1, if you want to train on the whole training dataset once per epoch,
        in which way the trainer will compute the iters_per_epoch based on the length of the dataset.
        But we don't recommend to do in this way, because the frame work is designed to train based
        on iteration, you should set it to the maximum training iterations you need.
        """
        "iters_per_epoch": 1000,

    }
}
```