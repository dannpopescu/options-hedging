import torch
import torch.nn as nn


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        dims = [32, 64]
        dims.insert(0, state_dim + action_dim)
        dims.append(1)

        initial_norm_fc = nn.BatchNorm1d(dims[0], momentum=0.01, eps=1e-3)

        self.q1 = Q(dims, initial_norm_fc)
        self.q2 = Q(dims, initial_norm_fc)

        self.device = self.q1.device

    def _format(self, state, action):
        x, u = state, action
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x,
                             device=self.device,
                             dtype=torch.float32)
            x = x.unsqueeze(0)
        if not isinstance(u, torch.Tensor):
            u = torch.tensor(u,
                             device=self.device,
                             dtype=torch.float32)
            u = u.unsqueeze(0)
        return torch.cat((x, u), dim=1)

    def forward(self, state, action):
        x = self._format(state, action)
        x1 = self.q1(x)
        x2 = self.q2(x)
        expected_reward = x1 - 1.5 * torch.sqrt(torch.clamp_min(x2 - x1*x1, 0) + 1e-15)
        return expected_reward

    def Q1(self, state, action):
        x = self._format(state, action)
        return self.q1(x)

    def Q2(self, state, action):
        x = self._format(state, action)
        return self.q2(x)

    def load(self, experiences):
        states, actions, rewards, new_states, is_terminals = experiences
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        new_states = torch.from_numpy(new_states).float().to(self.device)
        is_terminals = torch.from_numpy(is_terminals).float().to(self.device)
        return states, actions, rewards, new_states, is_terminals


class Q(nn.Module):
    def __init__(self, dims, initial_norm_fc):
        super(Q, self).__init__()

        self.initial_normalization = initial_norm_fc

        modules = []
        for i in range(1, len(dims)):
            modules.append(nn.Linear(dims[i-1], dims[i]))
            if i != (len(dims) - 1):
                modules.append(nn.ReLU())
                modules.append(nn.BatchNorm1d(dims[i], momentum=0.01, eps=1e-3))

        self.nn = nn.Sequential(*modules)

        self.nn.apply(init_weights)

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
        self.to(self.device)

    def forward(self, x):
        x = self.initial_normalization(x)
        return self.nn(x)