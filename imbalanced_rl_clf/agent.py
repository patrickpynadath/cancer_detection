import math
import random
from .utils import ReplayMemory
import torch
from torch.optim import Adam
from gymnasium import spaces
from .dqn import DQN
from torch import no_grad


class Agent:
    def __init__(self,
                 num_classes,
                 eps_end,
                 eps_start,
                 eps_decay,
                 encoder,
                 device,
                 mem_capacity,
                 batch_size,
                 lr):
        self.eps_end = eps_end
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.num_classes = num_classes
        self.encoder = encoder
        self.device = device
        self.policy_net = DQN(encoder).to(device)
        self.target_net = DQN(encoder).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.steps_done = 0
        self.mem = ReplayMemory(mem_capacity)
        self.batch_size = batch_size
        self.lr = lr
        self.action_space = spaces.Discrete(num_classes)
        self.optimizer = Adam(self.policy_net.mp.parameters(), lr=lr, eps=.001)

    def select_action(self, state_img):
        state_img = state_img.to(self.device)
        sample = random.random()
        eps_threshold = self.calc_eps_threshold()
        if sample > eps_threshold:
            with no_grad():
                return self.policy_net(state_img[None, :]).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[self.action_space.sample()]], device=self.device, dtype=torch.long)


    def calc_eps_threshold(self):
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * self.steps_done / self.eps_decay)
        return eps_threshold

    def remember_transitions(self):
        return self.mem.sample(self.batch_size)

    def get_optimizer(self):
        return self.optimizer

    def store_memory(self, state, action, next_state, reward):
        self.mem.push(state, action, next_state, reward)
        return

    def __len__(self):
        return len(self.mem)

    def get_batch_pred(self, batch):
        return torch.nn.functional.softmax(self.policy_net(batch))



