from enum import Enum

import torch


class Networks(Enum):
    MLP = 'mlp'
    ConvNet = 'convnet'
    LeNet = 'lenet'
    AlexNet = 'alexnet'


class Datasets(Enum):
    MNIST = 'mnist'
    FASHTION_MNIST = 'fashion_mnist'
    CIFAR10 = 'cifar10'
    SVHN = 'svhn'


LR_SYNTHETIC = 0.1
LR_NETWORK = 0.01

BATCH_SIZE_REAL = 256
BATCH_SIZE_TRAIN = 256
BATCH_SIZE_TEST = 256

ITERATIONS = 1000

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
