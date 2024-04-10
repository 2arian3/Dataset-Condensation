import torch
import torchvision
from torch.utils.data import DataLoader

from utils.common import get_network, DatasetInfo
from utils.consts import ITERATIONS, BATCH_SIZE_TEST, BATCH_SIZE_REAL


# Loading synthetic dataset
dataset = torch.load('syndata/CIFAR10_MLP_ipc-50_exp-0.pt')
imgs, labs = dataset['syn_imgs'], dataset['syn_labels']

# Using the actual dataset for testing
dataset = DatasetInfo('CIFAR10')

net = get_network('MLP', imgs[0].shape[0], 10, imgs[0].shape[1:3])

opt = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.5)
loss_fn = torch.nn.CrossEntropyLoss()

net.train()

for i in range(ITERATIONS):
    for data in DataLoader(torch.utils.data.TensorDataset(imgs, labs), batch_size=BATCH_SIZE_REAL, shuffle=True):
        img, lab = data
        out = net(img)

        loss = loss_fn(out, lab.long())

        print(f'LOSS: {loss.item()}')

        opt.zero_grad()
        loss.backward()
        opt.step()

# Evaluation of the trained model
net.eval()
correct = 0
total = 0
for data in torch.utils.data.DataLoader(dataset.test_dataset, batch_size=BATCH_SIZE_TEST, shuffle=False):
    img, lab = data
    out = net(img)
    _, predicted = torch.max(out.data, 1)
    total += lab.size(0)
    correct += (predicted == lab).sum().item()

print('Accuracy of the %s network on the %d test images: %d %%' % ('MLP', total, 100 * correct / total))
