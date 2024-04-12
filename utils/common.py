import time
import datetime
import logging
import os
import copy

from .consts import (Datasets,
                     Networks,
                     PWD,
                     BATCH_SIZE_TRAIN,
                     DEVICE)

from models.MultiLayerPerceptron import MultiLayerPerceptron
from models.ConvNet import ConvNet
from models.LeNet import LeNet
from models.AlexNet import AlexNet

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image


class Logger:
    LOG_PATH = f'{PWD}/logs/log_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'

    def __init__(self) -> None:
        logging.basicConfig(filename=Logger.LOG_PATH,
                        level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    
    @staticmethod
    def info(message):
        logging.info(message)


# This class is used to create a dataset from the synthetic images and labels and create a dataloader based on this dataset
class TensorDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]


# This class is used to get the information about the dataset
class DatasetInfo:
    DATASETS_INFO ={
        'mnist': {
            'img_size': (1, 28, 28),
            'num_of_classes': 10,
            'class_names': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
            'mean': (0.1307,),
            'std': (0.3081,)
        },
        'fashionmnist': {
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

        self.img_size = dataset_info['img_size'][1:3]
        self.num_of_classes = dataset_info['num_of_classes']
        self.class_names = dataset_info['class_names']
        self.num_of_channels = dataset_info['img_size'][0]

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


# This function is used to get the network based on the network name and the dataset information
def get_network(network_name: str, num_channels: int, num_classes: int, img_size=(28, 28)) -> torch.nn.Module | None:

    network_name = network_name.lower()

    if network_name == Networks.MLP.value:
        model = MultiLayerPerceptron(input_dim=img_size[0] * img_size[1] * num_channels, output_dim=num_classes)

    elif network_name == Networks.ConvNet.value:
        model = ConvNet(num_channels=num_channels, num_classes=num_classes, img_size=img_size)

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


# Used in loggings
def get_current_time():
    return datetime.datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')


# Hyperparameters for the outer and inner loops
def get_outer_and_inner_loops(ipc=1):
    outer_loop, inner_loop = 1, 1

    # These values are determined by the authors of the paper and supposed to work the best
    if ipc == 1:
        outer_loop, inner_loop = 1, 1

    elif ipc == 10:
        outer_loop, inner_loop = 10, 50

    elif ipc == 50:
        outer_loop, inner_loop = 50, 10

    return outer_loop, inner_loop


def get_eval_pool(model, eval_mode='M'):
    # Multi-Architecture Evaluation Pool
    if eval_mode == 'M':
        model_eval_pool = ['MLP', 'ConvNet', 'LeNet', 'AlexNet']

    elif eval_mode == 'I':
        model_eval_pool = [model]

    return model_eval_pool


def distance_wb(g_real, g_syn):
    shape = g_real.shape
    if len(shape) == 4:
        g_real = g_real.reshape(shape[0], shape[1] * shape[2] * shape[3])
        g_syn = g_syn.reshape(shape[0], shape[1] * shape[2] * shape[3])

    elif len(shape) == 3:
        g_real = g_real.reshape(shape[0], shape[1] * shape[2])
        g_syn = g_syn.reshape(shape[0], shape[1] * shape[2])

    elif len(shape) == 1:
        g_real = g_real.reshape(1, shape[0])
        g_syn = g_syn.reshape(1, shape[0])

        return torch.tensor(0, dtype=torch.float, device=DEVICE)

    d_weight = torch.sum(1 - torch.sum(g_real * g_syn, dim=-1) / (torch.norm(g_real, dim=-1) * torch.norm(g_syn, dim=-1) + 1e-6))
    d = d_weight

    return d


def get_match_loss(g_syn, g_real, metric='fancy'):
    d = torch.tensor(0.0).to(DEVICE)

    if metric == 'mse':
        g_real_vec = []
        g_syn_vec = []

        for ig in range(len(g_real)):
            g_real_vec.append(g_real[ig].reshape((-1)))
            g_syn_vec.append(g_syn[ig].reshape((-1)))

        g_real_vec = torch.cat(g_real_vec, dim=0)
        g_syn_vec = torch.cat(g_syn_vec, dim=0)
        d = torch.sum((g_syn_vec - g_real_vec)**2)

    elif metric == 'fancy':

        for ig in range(len(g_real)):
            gr = g_real[ig]
            gs = g_syn[ig]
            d += distance_wb(gr, gs)

    elif metric == 'cos':
        g_real_vec = []
        g_syn_vec = []

        for ig in range(len(g_real)):
            g_real_vec.append(g_real[ig].reshape((-1)))
            g_syn_vec.append(g_syn[ig].reshape((-1)))

        g_real_vec = torch.cat(g_real_vec, dim=0)
        g_syn_vec = torch.cat(g_syn_vec, dim=0)
        d = 1 - torch.sum(g_real_vec * g_syn_vec, dim=-1) / (torch.norm(g_real_vec, dim=-1) * torch.norm(g_syn_vec, dim=-1) + 1e-6)

    return d


# This function is used to get random images from the real dataset
def get_random_images(real_imgs, class_indices, c, num_imgs=1):
    shuffle_indices = np.random.permutation(class_indices[c])[:num_imgs]
    return real_imgs[shuffle_indices]


# An iteration is a a single pass through the dataset which may be used for training (updating weights) or evaluation
def iteration(net, loss_fn, optimizer, train_loader, is_training=True):
    loss_avg, acc_avg, num_of_exp = 0, 0, 0
    net = net.to(DEVICE)
    loss_fn = loss_fn.to(DEVICE)

    if is_training:
        net.train()
    else:
        net.eval()

    for _, data in enumerate(train_loader):
        img = data[0].float().to(DEVICE)
        lab = data[1].long().to(DEVICE)
        n_b = lab.shape[0]

        out = net(img)
        l = loss_fn(out, lab)
        acc = np.sum(np.equal(np.argmax(out.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

        loss_avg += l.item() * n_b
        acc_avg += acc
        num_of_exp += n_b

        if is_training:
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
    
    loss_avg /= num_of_exp
    acc_avg /= num_of_exp

    return loss_avg, acc_avg


def get_synset_evaluation(it_eval, net_eval, image_syn_eval, label_syn_eval, test_loader, epoch_eval_train, lr_network):
    image_syn_eval, label_syn_eval = image_syn_eval.to(DEVICE), label_syn_eval.to(DEVICE)

    lr = lr_network
    lr_scheduler = [epoch_eval_train // 2 + 1]

    optimizer = torch.optim.SGD(net_eval.parameters(), lr=lr, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss().to(DEVICE)

    train_dataset = TensorDataset(image_syn_eval, label_syn_eval)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True)

    start_time = time.time()
    for epoch in range(epoch_eval_train + 1):
        train_loss, train_accuracy = iteration(net_eval, loss_fn, optimizer, train_loader, is_training=True)
        if epoch in lr_scheduler:
            lr *= 0.1
            optimizer = torch.optim.SGD(net_eval.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    finish_time = time.time() - start_time
    test_loss, test_accuracy = iteration(net_eval, loss_fn, optimizer, test_loader, is_training=False)

    print('Evaluation for iteration {} is completed in {:.2f} seconds, train loss = {:.2f}, train acc = {:.2f}, test loss = {:.2f}, test acc = {:.2f}'.format(it_eval, finish_time, train_loss, train_accuracy, test_loss, test_accuracy))

    return train_accuracy, test_accuracy


# Visualizing the synthetic images
def save_img_result(img_syn, args, exp, it, dataset_info):
    save_path = os.path.join(PWD, 'syndata', f'{args.network}_{args.dataset}_{args.ipc}_exp{exp}_it{it}.png')
    img_syn_visualizer = copy.deepcopy(img_syn.detach().cpu())

    for channel in range(dataset_info.num_of_channels):
        img_syn_visualizer[:, channel] = img_syn_visualizer[:, channel] * dataset_info.std[channel] + dataset_info.mean[channel]

    img_syn_visualizer[img_syn_visualizer < 0] = 0.0
    img_syn_visualizer[img_syn_visualizer > 1] = 1.0

    save_image(img_syn_visualizer, save_path, nrow=args.ipc)
