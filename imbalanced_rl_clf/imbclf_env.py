import gymnasium as gym
from gymnasium import spaces
import numpy as np


class ImbalancedClfEnv(gym.Env):

    def __init__(self, dataset, device):
        super(ImbalancedClfEnv, self).__init__()

        self.dataset = dataset
        self.step_id = 0
        self.cur_idx = None
        self.lmbda = self._calc_lambda()
        self.device = device
        self.running_reward = 0
        self.num_pos_total = 0
        self.num_pos_right = 0
        self.observation_space = spaces.Discrete(len(dataset))
        self.mask = np.ones(len(dataset), dtype=np.int8)
        self.rewards = []

    def reset(self, **kwargs):
        self.mask = np.ones(shape=(len(self.dataset),), dtype=np.int8)
        self.running_reward = 0
        self.cur_idx = int(self.observation_space.sample(mask=self.mask))
        self.num_pos_total = 0
        self.num_pos_right = 0
        self.rewards = []
        return self.cur_idx, {}

    # function for actually getting the image
    def get_sample(self, sample_id):
        return self.dataset[int(sample_id)][0].to(self.device), self.dataset[sample_id][1].to(self.device)

    def step(self, action):
        # making sure that this sample doesn't get drawn again
        # the correction prediction
        env_action = int(self.dataset[self.cur_idx][-1])
        terminated = not np.any(self.mask)
        if env_action == 1:
            self.num_pos_total += 1
            if action == env_action:
                reward = 1
                self.num_pos_right += 1
            else:
                terminated = True
                reward = -10
        else:
            if action == env_action:
                reward = self.lmbda
            else:
                reward = -1 * self.lmbda

        self.rewards.append(reward)
        state = self._sample_state()
        return state, reward, terminated, {}

    def _sample_state(self):
        idx = self.observation_space.sample(mask = self.mask)
        self.mask[idx] = 0 # no repeats during an episode
        self.cur_idx = idx
        return int(self.cur_idx)

    def _calc_lambda(self):
        num_pos = 0
        num_neg = 0
        for i in range(len(self.dataset)):
            if self.dataset[i][2] == 1:
                num_pos += 1
            else:
                num_neg += 1
        return num_pos / num_neg

    def get_reward_hist(self):
        return np.array(self.rewards)
