import datetime

from .consts import Datasets, Networks, DEVICE

from models.MultiLayerPerceptron import MultiLayerPerceptron
from models.ConvNet import ConvNet
from models.LeNet import LeNet
from models.AlexNet import AlexNet

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class DatasetInfo:
    DATASETS_INFO ={
        'mnist': {
            'img_size': (1, 28, 28),
            'num_of_classes': 10,
            'class_names': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
            'mean': (0.1307,),
            'std': (0.3081,)
        },
        'fashion_mnist': {
            'img_size': (1, 28, 28),
            'num_of_classes': 10,
            'class_names': ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
            'mean': (0.2860,),
            'std': (0.3530,)
        },
        'cifar10': {
            'img_size': (3, 32, 32),
            'num_of_classes': 10,
            'class_names': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
            'mean': (0.4914, 0.4822, 0.4465),
            'std': (0.2023, 0.1994, 0.2010)
        },
        'svhn': {
            'img_size': (3, 32, 32),
            'num_of_classes': 10,
            'class_names': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
            'mean': (0.4377, 0.4438, 0.4728),
            'std': (0.1980, 0.2010, 0.1970)
        }
    }
    def __init__(self, dataset_name: str, dataset_path='./data') -> None:
        dataset_name = dataset_name.lower()

        dataset_info = DatasetInfo.DATASETS_INFO[dataset_name]

        self.img_size = dataset_info['img_size']
        self.num_of_classes = dataset_info['num_of_classes']
        self.class_names = dataset_info['class_names']
        self.num_of_channels = self.img_size[0]

        self.mean = dataset_info['mean']
        self.std = dataset_info['std']

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])


        if dataset_name == Datasets.MNIST.value:
            self.train_dataset = datasets.MNIST(root=dataset_path, train=True, download=True, transform=self.transform)
            self.test_dataset = datasets.MNIST(root=dataset_path, train=False, download=True, transform=self.transform)

        elif dataset_name == Datasets.FASHTION_MNIST.value:
            self.train_dataset = datasets.FashionMNIST(root=dataset_path, train=True, download=True, transform=self.transform)
            self.test_dataset = datasets.FashionMNIST(root=dataset_path, train=False, download=True, transform=self.transform)

        elif dataset_name == Datasets.CIFAR10.value:
            self.train_dataset = datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=self.transform)
            self.test_dataset = datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=self.transform)

        elif dataset_name == Datasets.SVHN.value:
            self.train_dataset = datasets.SVHN(root=dataset_path, split='train', download=True, transform=self.transform)
            self.test_dataset = datasets.SVHN(root=dataset_path, split='test', download=True, transform=self.transform)


class TensorDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]


def get_network(network_name: str, num_channels: int, num_classes: int, img_size=(28, 28)):

    network_name = network_name.lower()

    if network_name == Networks.MLP.value:
        model = MultiLayerPerceptron(input_dim=img_size[0] * img_size[1] * num_channels, output_dim=num_classes)

    elif network_name == Networks.ConvNet.value:
        pass

    elif network_name == Networks.LeNet.value:
        model = LeNet(num_channels=num_channels, num_classes=num_classes)

    elif network_name == Networks.AlexNet.value:
        model = AlexNet(num_channels=num_channels, num_classes=num_classes)

    else:
        return None
    
    gpu_instances = torch.cuda.device_count()

    if gpu_instances > 1:
        model = torch.nn.DataParallel(model)

    model = model.to(DEVICE)

    return model


def get_current_time():
    return datetime.datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')


def get_random_imgs(imgs, class_indices, num_imgs=1):
    random_imgs = []

    for c in range(len(class_indices)):
        random_indices = torch.randperm(len(class_indices[c]))[:num_imgs]
        random_imgs.append(imgs[class_indices[c][random_indices]])

    return random_imgs
