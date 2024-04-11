import torch.nn as nn
import torch.nn.functional as F


class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(MultiLayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)


    def forward(self, x, apply_softmax=False):
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        if apply_softmax:
            x = F.softmax(x, dim=1)

        return x