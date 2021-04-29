import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 action_bounds):
        super(Actor, self).__init__()
        self.env_min, self.env_max = action_bounds

        hidden_dims = (32, 64)

        self.nn = nn.Sequential(
            nn.BatchNorm1d(state_dim),
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.Linear(hidden_dims[1], action_dim),
            nn.Sigmoid()
        )

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
        self.to(self.device)

        self.env_min = torch.tensor(self.env_min,
                                    device=self.device,
                                    dtype=torch.float32)

        self.env_max = torch.tensor(self.env_max,
                                    device=self.device,
                                    dtype=torch.float32)

        self.rescale_fn = lambda x: x * self.env_max

    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x,
                             device=self.device,
                             dtype=torch.float32)
            x = x.unsqueeze(0)
        return x

    def forward(self, state):
        x = self._format(state)
        x = self.nn(x)
        return self.rescale_fn(x)
