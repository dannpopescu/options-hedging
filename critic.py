import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()

        self.input_layer = nn.Sequential(nn.BatchNorm1d(input_dim))
        self.hidden_layer1 = nn.Sequential(nn.Linear(input_dim, 64, bias=False),
                                           nn.BatchNorm1d(64),
                                           nn.ReLU())
        self.hidden_layer2 = nn.Sequential(nn.Linear(64, 64, bias=False),
                                           nn.BatchNorm1d(64),
                                           nn.ReLU())
        self.output_layer = nn.Sequential(nn.Linear(64, 1))

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
        self.to(self.device)

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
        return x, u

    def forward(self, state, action):
        x, u = self._format(state, action)
        x = torch.cat((x, u), dim=1)
        x = self.input_layer(x)
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        return self.output_layer(x)

    def load(self, experiences):
        states, actions, rewards, new_states, is_terminals = experiences
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        new_states = torch.from_numpy(new_states).float().to(self.device)
        is_terminals = torch.from_numpy(is_terminals).float().to(self.device)
        return states, actions, rewards, new_states, is_terminals