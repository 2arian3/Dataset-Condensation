import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(MultiLayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)


    def forward(self, x_in, apply_softmax=False):
        y_pred = F.relu(self.fc1(x_in))
        y_pred = F.relu(self.fc2(y_pred))
        y_pred = self.fc2(y_pred)

        if apply_softmax:
            y_pred = F.softmax(y_pred, dim=1)

        return y_pred