import torch.nn as nn


class DQN(nn.Module):

    def __init__(self, encoder):
        super(DQN, self).__init__()
        self.encoder = encoder

        self.mp = nn.Sequential(nn.Linear(1024, 512),
                                nn.ReLU(),
                                nn.LazyBatchNorm1d(),
                                nn.Linear(512, 256),
                                nn.ReLU(),
                                nn.LazyBatchNorm1d(),
                                nn.Linear(256, 2),
                                nn.ReLU())

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        z = self.encoder(x, None)
        return self.mp(z)