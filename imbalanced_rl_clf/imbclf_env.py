import gymnasium as gym
from gymnasium import spaces
import numpy as np


class ImbalancedClfEnv(gym.Env):

    def __init__(self, dataset, device):
        super(ImbalancedClfEnv, self).__init__()

        self.dataset = dataset
        self.cur_state = np.ones(shape=(len(dataset),), dtype=np.int8)
        self.step_id = 0
        self.cur_idx = None
        self.lmbda = self._calc_lambda()
        self.device = device
        self.running_reward = 0
        self.num_pos_total = 0
        self.num_pos_right = 0
        self.observation_space = spaces.Discrete(len(dataset))
        self.mask = np.ones(len(dataset))

    def reset(self, **kwargs):
        self.cur_state = np.ones(shape=(len(self.dataset),), dtype=np.int8)
        self.running_reward = 0
        self.cur_idx = self.observation_space.sample(mask=self.cur_state)
        self.num_pos_total = 0
        self.num_pos_right = 0
        return self.cur_idx, {}

    # function for actually getting the image
    def get_sample(self, sample_id):
        return self.dataset[int(sample_id)][0].to(self.device), self.dataset[sample_id][1].to(self.device)

    def step(self, action):
        # making sure that this sample doesn't get drawn again
        self.cur_state[self.cur_idx] = 0
        # the correction prediction
        env_action = self.dataset[self.cur_idx][-1]
        done = False
        if action == env_action:
            if env_action == 1:
                reward = 1
                self.num_pos_total += 1
                self.num_pos_right += 1
            else:
                reward = self.lmbda
        else:
            if env_action == 1:
                done = True
                self.num_pos_total += 1
                reward = -1
            else:
                reward = -self.lmbda
        self.running_reward += reward
        state = self._sample_state()
        return state, reward, done, {}

    def _sample_state(self):
        idx = self.observation_space.sample(mask = self.mask)
        self.mask[idx] = 0 # no repeats during an episode
        return idx

    def _calc_lambda(self):
        num_pos = 0
        num_neg = 0
        for i in range(len(self.dataset)):
            if self.dataset[i][2] == 1:
                num_pos += 1
            else:
                num_neg += 1
        return num_pos / num_neg