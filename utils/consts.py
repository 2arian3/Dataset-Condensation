from enum import Enum


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