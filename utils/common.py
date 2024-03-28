from consts import Datasets, Networks

from ..models.MultiLayerPerceptron import MultiLayerPerceptron
from ..models.ConvNet import ConvNet
from ..models.LeNet import LeNet
from ..models.AlexNet import AlexNet

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
    def __init__(self, dataset_name) -> None:
        dataset_info = DatasetInfo.DATASETS_INFO[dataset_name]

        self.img_size = dataset_info['img_size']
        self.num_of_classes = dataset_info['num_of_classes']
        self.class_names = dataset_info['class_names']

        self.mean = dataset_info['mean']
        self.std = dataset_info['std']

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

        if dataset_name == Datasets.MNIST:
            self.train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=self.transform)
            self.test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=self.transform)

        elif dataset_name == Datasets.FASHTION_MNIST:
            self.train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=self.transform)
            self.test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=self.transform)

        elif dataset_name == Datasets.CIFAR10:
            self.train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform)
            self.test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=self.transform)

        elif dataset_name == Datasets.SVHN:
            self.train_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=self.transform)
            self.test_dataset = datasets.SVHN(root='./data', split='test', download=True, transform=self.transform)



class TensorDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]
    


def get_network(network_name, num_channels, num_classes, img_size=(28, 28)):
    if network_name == Networks.MLP:
        pass

    elif network_name == Networks.ConvNet:
        pass

    elif network_name == Networks.LeNet:
        pass

    elif network_name == Networks.AlexNet:
        pass


