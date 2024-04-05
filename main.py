import os
import copy
import torch
import numpy as np
import argparse

from utils.common import (DatasetInfo,
                          TensorDataset,
                          iteration,
                          get_network,
                          get_current_time,
                          get_match_loss,
                          get_eval_pool,
                          get_random_images,
                          get_synset_evaluation,
                          get_outer_and_inner_loops)

from utils.consts import (DEVICE,
                          ITERATIONS,
                          LR_NETWORK,
                          LR_SYNTHETIC,
                          BATCH_SIZE_REAL,
                          BATCH_SIZE_TEST,
                          BATCH_SIZE_TRAIN)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process the input parameters')
    
    parser.add_argument('--dataset', type=str, default='MNIST', help='Dataset to use')
    parser.add_argument('--network', type=str, default='MLP', help='Network to use')
    parser.add_argument('--ipc', type=int, default=1, help='Images per class for the sythetic dataset')
    parser.add_argument('--num_of_exp', type=int, default=5, help='Number of experiments to run')
    parser.add_argument('--num_of_eval', type=int, default=20, help='Number of evaluations to run')
    parser.add_argument('--epoch_eval_train', type=int, default=300, help='Number of epochs to evaluate the training')
    parser.add_argument('--syn_init', type=str, default='random', help='Synthetic dataset initialization')

    args = parser.parse_args()


    if not os.path.exists('data'):
        os.makedirs('data')


    if not os.path.exists('syndata'):
        os.makedirs('syndata')


    dataset = DatasetInfo(dataset_name=args.dataset)


    outer_loop, inner_loop = get_outer_and_inner_loops(args.ipc)
    model_eval_pool = get_eval_pool(args.network, eval_mode='I')
    it_eval_pool = np.arange(0, ITERATIONS+1, 500).tolist()

    data_to_save = []
    
    exps_records = {}
    for key in model_eval_pool:
        exps_records[key] = []


    test_loader = torch.utils.data.DataLoader(dataset.test_dataset, batch_size=BATCH_SIZE_TEST, shuffle=False)


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

        # Synthetic dataset initialization if the mode is 'real'
        if args.syn_init == 'real':
            for c in range(dataset.num_of_classes):
                syn_imgs.data[c*args.ipc:(c+1)*args.ipc] = get_random_images(real_imgs, class_indices, c, args.ipc).detach().data

        img_opt = torch.optim.SGD([syn_imgs], lr=LR_SYNTHETIC, momentum=0.5)
        img_opt.zero_grad()
        loss_fn = torch.nn.CrossEntropyLoss().to(DEVICE)

        print(f'Training begins at {get_current_time()}')

        for it in range(ITERATIONS + 1):

            if it in it_eval_pool:
                for model_eval in model_eval_pool:
                    print(f'\n***Evaluation for {model_eval} at iteration {it+1}/{ITERATIONS}***\n')

                    accuracies = []
                    
                    for it_eval in range(args.num_of_eval):
                        net_eval = get_network(model_eval, dataset.num_of_channels, dataset.num_of_classes, dataset.img_size)
                        image_syn_eval, label_syn_eval = copy.deepcopy(syn_imgs.detach()), copy.deepcopy(syn_labels.detach())
                        # TODO: evaluate synthetic dataset
                        train_acc, test_acc = get_synset_evaluation(it_eval, net_eval, image_syn_eval, label_syn_eval, test_loader, args.epoch_eval_train)
                        accuracies.append(test_acc)

                    if it == ITERATIONS:
                        exps_records[model_eval].append(accuracies)
            
            net = get_network(args.network, dataset.num_of_channels, dataset.num_of_classes, dataset.img_size)
            net.train()

            net_params = list(net.parameters())

            net_opt = torch.optim.SGD(net.parameters(), lr=LR_NETWORK, momentum=0.5)
            net_opt.zero_grad()

            loss_avg = 0

            for outer in range(outer_loop):

                # TODO: Check if the model contains a BatchNorm layer
                
                loss = torch.tensor(0.0).to(DEVICE)
                
                for c in range(dataset.num_of_classes):
                    real_img = get_random_images(real_imgs, class_indices, c, BATCH_SIZE_REAL)
                    real_lab = torch.ones((real_img.shape[0],), dtype=torch.long).to(DEVICE) * c
                    syn_img = syn_imgs[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc,
                                                                           dataset.num_of_channels,
                                                                           dataset.img_size[0], dataset.img_size[1]))
                    syn_lab = torch.ones((args.ipc,), dtype=torch.long).to(DEVICE) * c

                    real_output = net(real_img)
                    real_loss = loss_fn(real_output, real_lab)

                    g_real = torch.autograd.grad(real_loss, net_params)
                    g_real = list((_.detach().clone() for _ in g_real))

                    syn_output = net(syn_img)
                    syn_loss = loss_fn(syn_output, syn_lab)
                    g_syn = torch.autograd.grad(syn_loss, net_params, create_graph=True)

                    loss += get_match_loss(g_syn, g_real)
                
                img_opt.zero_grad()
                loss.backward()
                img_opt.step()
                loss_avg += loss.item()

                # if outer == outer_loop - 1:
                #     break
                
                syn_img_train, syn_lab_train = copy.deepcopy(syn_imgs.detach()), copy.deepcopy(syn_labels.detach())
                syn_train_dataset = TensorDataset(syn_img_train, syn_lab_train)
                train_loader = torch.utils.data.DataLoader(syn_train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True)

                for inner in range(inner_loop):
                    inner_loss_avg, inner_acc_avg = iteration(net, loss_fn, net_opt, train_loader, is_training=True)

                loss_avg /= (dataset.num_of_classes * outer_loop)

                if it%10 == 0:
                    print(f'Iteration {it+1}/{ITERATIONS+1}, Outer loop {outer+1}/{outer_loop}, Inner loop {inner+1}/{inner_loop}, Time {get_current_time()}, Loss {loss_avg}, Inner Loss {inner_loss_avg}, Inner Acc {inner_acc_avg}')
                
                if it == ITERATIONS:
                    data_to_save.append([copy.deepcopy(syn_imgs.detach().cpu()), copy.deepcopy(syn_labels.detach().cpu())])
                    torch.save({'syn_imgs': syn_imgs, 'syn_labels': syn_labels}, f'syndata/{args.dataset}_{args.network}_ipc-{args.ipc}_exp-{exp}.pt')


    print('***Finished Training***')
    for key in model_eval_pool:
        accuracies = exps_records[key]
        print(f'Run {args.num_of_exp} experiments, train on {args.network}, evaluate {len(accuracies)} random {key}, mean  = {np.mean(accuracies)*100}, std = {np.std(accuracies)*100}')
