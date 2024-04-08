from enum import Enum

import torch
import os


class Networks(Enum):
    MLP = 'mlp'
    ConvNet = 'convnet'
    LeNet = 'lenet'
    AlexNet = 'alexnet'


class Datasets(Enum):
    MNIST = 'mnist'
    FASHTION_MNIST = 'fashionmnist'
    CIFAR10 = 'cifar10'
    SVHN = 'svhn'


BATCH_SIZE_REAL = 256
BATCH_SIZE_TRAIN = 256
BATCH_SIZE_TEST = 256

ITERATIONS = 1000

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

PWD = os.getcwd().replace(' ', '\ ')