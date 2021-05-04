import torch
import numpy as np


class GreedyStrategy():
    def select_action(self, model, state):
        return get_greedy_action(model, state)


class NormalNoiseStrategy():
    def __init__(self, epsilon, min_epsilon, epsilon_decay, action_bounds, noise_scale, noise_clip):
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay

        self.low, self.high = action_bounds
        self.noise_scale = noise_scale
        self.noise_clip = noise_clip

    def epsilon_update(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def select_action(self, model, state, env):
        action = get_greedy_action(model, state)
        is_exploratory = False

        if np.random.rand() <= self.epsilon_decay:
            noise = np.clip(np.random.normal(0, self.noise_scale), -self.noise_clip, self.noise_clip)
            noisy_action = action + noise
            action = np.clip(noisy_action, self.low, self.high)[0]
            is_exploratory = True

        return action, is_exploratory


class EGreedyExpStrategy():
    def __init__(self, epsilon=1.0, min_epsilon=0.1, epsilon_decay=0.9999):
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
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