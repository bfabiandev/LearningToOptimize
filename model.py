import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class MnistModel(nn.Module):

    def __init__(self):
        super(MnistModel, self).__init__()
        self.linear1 = nn.Linear(28 * 28, 32)
        self.linear2 = nn.Linear(32, 10)

    def forward(self, inputs):
        x = inputs.view(-1, 28 * 28)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return F.log_softmax(x, dim=1)


class LinearRegressionModel(nn.Module):

    def __init__(self, dim=10):
        super(LinearRegressionModel, self).__init__()
        self.dim = dim
        self.linear1 = nn.Linear(dim, 1)

    def forward(self, inputs):
        x = inputs.view(-1, self.dim)
        return F.sigmoid(self.linear1(x))
