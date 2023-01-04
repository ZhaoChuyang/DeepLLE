from setuptools import setup, find_packages

setup(
    name='DeepLLE',
    version='0.0.1',
    description='Distributed PyTorch Training Framework for Low-light Enhancement',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
    ],
)