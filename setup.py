from setuptools import setup, find_packages

setup(
    name='deeplle',
    version='1.0.0',
    author="Intel",
    description='Distributed PyTorch Training Framework for Low-light Enhancement',
    packages=find_packages(exclude=("configs", "tests*")),
    python_requires='>=3.7',
    install_requires=[
        "torch>=1.7.0",
        "torchvision>=0.8.0",
        "pandas",
        "tensorboard",
        "tqdm",
        "termcolor",
    ]
)