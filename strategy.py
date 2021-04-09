import torch
import numpy as np


class GreedyStrategy():
    def __init__(self, bounds):
        self.low, self.high = bounds
        self.ratio_noise_injected = 0

    def select_action(self, model, state):
        model.eval()
        with torch.no_grad():
            greedy_action = model(state).cpu().detach().data.numpy().squeeze()
        model.train()

        action = np.clip(greedy_action, self.low, self.high)
        return np.reshape(action, self.high.shape)


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
