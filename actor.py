import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 action_bounds):
        super(Actor, self).__init__()
        self.env_min, self.env_max = action_bounds

        self.input_layer = nn.Sequential(nn.BatchNorm1d(input_dim))
        self.hidden_layer1 = nn.Sequential(nn.Linear(input_dim, 64, bias=False),
                                           nn.BatchNorm1d(64),
                                           nn.ReLU())
        self.hidden_layer2 = nn.Sequential(nn.Linear(64, 64, bias=False),
                                           nn.BatchNorm1d(64),
                                           nn.ReLU())
        self.output_layer = nn.Sequential(nn.Linear(64, output_dim),
                                          nn.Tanh())

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

        self.nn_min = nn.Tanh()(torch.Tensor([float('-inf')])).to(self.device)
        self.nn_max = nn.Tanh()(torch.Tensor([float('inf')])).to(self.device)

        self.rescale_fn = lambda x: (x - self.nn_min) * (self.env_max - self.env_min) / \
                                    (self.nn_max - self.nn_min) + self.env_min

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
        x = self.input_layer(x)
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.output_layer(x)
        return self.rescale_fn(x)