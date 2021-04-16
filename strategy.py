import torch
import numpy as np


class GreedyStrategy():
    def select_action(self, model, state):
        return get_greedy_action(model, state)


class NormalNoiseStrategy():
    def __init__(self, bounds, exploration_noise_ratio=0.1):
        self.low, self.high = bounds
        self.exploration_noise_ratio = exploration_noise_ratio
        self.ratio_noise_injected = 0
        self.actions_nn = []
        self.actions_n = []
        self.actions_r = []

    def select_action(self, model, state, max_exploration=False):
        if max_exploration:
            noise_scale = 0.2
        else:
            noise_scale = self.exploration_noise_ratio * self.high

        model.eval()
        with torch.no_grad():
            greedy_action = model(state).cpu().detach().data.numpy().squeeze()
        model.train()
        self.actions_nn.append(greedy_action)

        noise = np.random.normal(loc=0, scale=noise_scale, size=len(self.high))
        self.actions_n.append(noise)

        noisy_action = greedy_action + noise

        action = np.clip(noisy_action, self.low, self.high)
        self.actions_r.append(action)

        self.ratio_noise_injected = np.mean(abs((greedy_action - action ) /(self.high - self.low)))
        return action[0]


class EGreedyExpStrategy():
    def __init__(self, init_epsilon=1.0, min_epsilon=0.1, epsilon_decay=0.9999):
        self.init_epsilon = init_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon = init_epsilon
        self.epsilon_decay = epsilon_decay

    def epsilon_update(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def select_action(self, model, state, env):
        is_exploratory = False
        if np.random.rand() > self.epsilon:
            action = get_greedy_action(model, state)
        else:
            action = env.action_space.sample()[0]
            is_exploratory = True

        return action, is_exploratory


def get_greedy_action(model, state):
    model.eval()
    with torch.no_grad():
        greedy_action = model(state).detach().cpu().data.numpy()[0][0]
    model.train()
    return greedy_action