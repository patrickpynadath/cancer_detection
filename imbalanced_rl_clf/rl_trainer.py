from .imbclf_env import ImbalancedClfEnv
from .agent import Agent
from .utils import Transition
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import matplotlib
from itertools import count
from torch.utils.tensorboard import SummaryWriter
import datetime


def timestamp():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


class RLTrainer:
    def __init__(self,
                 gamma,
                 tau : float,
                 env : ImbalancedClfEnv,
                 agent: Agent,
                 device : str,
                 log_dir = 'lightning_logs'):
        self.env = env
        self.device = device
        self.agent = agent
        self.gamma = gamma
        self.episode_durations = []
        self.tau = tau
        #self.logger = SummaryWriter(log_dir=f"{log_dir}/rl_net_{timestamp()}")


    def optimize_model(self):
        transitions = self.agent.remember_transitions()

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        non_final_next_states = torch.cat([self.env.get_sample(int(s))[1][None, :] for s in non_final_next_states])

        state_batch = torch.cat([self.env.get_sample(int(s))[1][None, :] for s in batch.state])

        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.agent.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.agent.batch_size, device=self.device)

        with torch.no_grad():
            next_state_values[non_final_mask] = self.agent.target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        # Optimize the model
        optimizer = self.get_agent_optimizer()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    def get_agent_optimizer(self):
        return self.agent.get_optimizer()

    def timestep(self, state, current_timestep_count):
        env = self.env
        agent = self.agent
        state_img = env.get_sample(state)[1]
        action = agent.select_action(state_img)
        observation, reward, terminated, truncated = env.step(action.item())

        reward = torch.tensor([reward], device=self.device)
        done = terminated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Store the transition in memory
        agent.store_memory(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        if len(agent) >= agent.batch_size:
            self.optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = agent.target_net.state_dict()
        policy_net_state_dict = agent.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        agent.target_net.load_state_dict(target_net_state_dict)
        if current_timestep_count % 25 == 0:
            print(f"{current_timestep_count} : {env.running_reward}, cur_state: {state}")
            if env.num_pos_total > 0:
                print(f"{current_timestep_count} : {env.num_pos_right / env.num_pos_total}")
        if done:
            self.episode_durations.append(current_timestep_count + 1)
            #self.logger.add_scalar('duration', scalar_value=current_timestep_count+1, global_step=1)
            #self.plot_durations()
        return done

    def plot_durations(self, show_result=False):
        is_ipython = 'inline' in matplotlib.get_backend()
        if is_ipython:
            from IPython import display
        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())

    def train_loop(self, model_updates):
        episode = 0
        model_update_val = 0
        while model_update_val < model_updates:
            print(f"starting episode {episode}")
            state, info = self.env.reset()
            state = int(state)
            for time_step in count():
                is_done = self.timestep(state, time_step)
                if is_done:
                    episode += 1
                    break
        print('Complete')
        torch.save(self.agent.policy_net.mp.state_dict(), 'rl_agent_policynet.pth')
        self.plot_durations(show_result=True)
        plt.ioff()
        plt.show()
