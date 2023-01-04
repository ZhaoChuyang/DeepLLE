from deeplle.engine.launch import launch
from deeplle.data.samplers import *
from deeplle.utils import comm


def test_training_sampler():
    training_sampler = TrainingSampler(20, True)
    training_sampler_iter = iter(training_sampler)
    indices = []
    for _ in range(5):
        indices.append(next(training_sampler_iter))
    
    indices = comm.gather(indices)
    if comm.is_main_process():
        print(indices)


def test_balanced_sampler():
    training_sampler = BalancedSampler([5, 2, 5, 5])
    training_sampler_iter = iter(training_sampler)
    indices = []
    for _ in range(5):
        indices.append(next(training_sampler_iter))
    
    indices = comm.gather(indices)
    if comm.is_main_process():
        print(indices)


def test_inference_sampler():
    training_sampler = InferenceSampler(13)
    indices = [i for i in training_sampler]
    
    indices = comm.gather(indices)
    if comm.is_main_process():
        print(indices)


def main():
    test_inference_sampler()
    

if __name__ == '__main__':
    args = ()
    launch(main, 4, 1, 0, dist_url='auto', args=args)

