log_dir=$1

tensorboard --port=8080 --logdir=$log_dir --host=0.0.0.0
