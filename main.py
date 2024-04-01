import os
import torch
import numpy as np
import argparse

from utils.common import (DatasetInfo,
                          TensorDataset,
                          get_network,
                          get_current_time,
                          get_random_imgs)

from utils.consts import (DEVICE,
                          LR_NETWORK,
                          LR_SYNTHETIC,
                          BATCH_SIZE_REAL,
                          BATCH_SIZE_TRAIN)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process the input parameters')
    
    parser.add_argument('--dataset', type=str, default='MNIST', help='Dataset to use')
    parser.add_argument('--network', type=str, default='MLP', help='Network to use')
    parser.add_argument('--ipc', type=int, default=1, help='Images per class for the sythetic dataset')
    parser.add_argument('--num_of_exp', type=int, default=5, help='Number of experiments to run')

    args = parser.parse_args()


    if not os.path.exists('data'):
        os.makedirs('data')


    if not os.path.exists('syndata'):
        os.makedirs('syndata')


    dataset = DatasetInfo(dataset_name=args.dataset)


    for exp in range(args.num_of_exp):

        print(f'Experiment {exp + 1}/{args.num_of_exp}')
        print(f'Hyperparameters: {args.__dict__}')

        real_imgs = [torch.unsqueeze(train_data[0], dim=0) for train_data in dataset.train_dataset]
        real_labels = [train_data[1] for train_data in dataset.train_dataset]
        class_indices = [torch.where(torch.tensor(real_labels) == i)[0] for i in range(dataset.num_of_classes)]

        real_imgs = torch.cat(real_imgs, dim=0).to(DEVICE)
        real_labels = torch.tensor(real_labels).to(DEVICE)

        for c in range(dataset.num_of_classes):
            print(f'Class {c}: {len(class_indices[c])} images \n')
        
        for c in range(dataset.num_of_channels):
            print(f'Channel {c+1}: mean={torch.mean(real_imgs[:, c])}, std={torch.std(real_imgs[:, c])}')

        syn_imgs = torch.randn(size=(dataset.num_of_classes * args.ipc, dataset.num_of_channels, dataset.img_size[0], dataset.img_size[1]),
                               dtype=torch.float, requires_grad=True, device=DEVICE)
        syn_labels = torch.tensor(np.array([np.ones(args.ipc) * i for i in range(dataset.num_of_classes)])).flatten().to(DEVICE)

        img_opt = torch.optim.SGD([syn_imgs], lr=LR_SYNTHETIC, momentum=0.5)
        img_opt.zero_grad()
        loss_fn = torch.nn.CrossEntropyLoss().to(DEVICE)

        print(f'Training begins at {get_current_time()}')
