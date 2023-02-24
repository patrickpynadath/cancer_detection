import torch.nn as nn
import torch


class DQN(nn.Module):

    def __init__(self, encoder):
        super(DQN, self).__init__()
        self.encoder = encoder


        self.l1 = nn.Linear(1024, 512)
        self.l2 = nn.Linear(512, 256)
        self.l4 = nn.Linear(256, 2)
        self.mp = nn.Sequential(self.l1, nn.ReLU(), self.l2, nn.ReLU(), self.l3, nn.ReLU(), self.l4)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        z = self.encoder(x, None)
        return self.mp(z)
