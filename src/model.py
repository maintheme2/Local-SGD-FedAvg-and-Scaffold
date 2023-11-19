import torch
from torch import nn
import torch.nn.functional as F


class LinearModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hidden_layers_num=1, apply_softmax=False):
        super(LinearModel, self).__init__()

        if hidden_layers_num < 0:
            print(f"Hidden layers num {hidden_layers_num} is less than 0. Resetting to 0...")
            hidden_layers_num = 0

        self.input_layer = nn.Linear(input_dim, hidden_dim)

        self.hidden_layers = nn.Sequential(
            *[nn.Linear(hidden_dim, hidden_dim)
              for _ in range(hidden_layers_num)]
        ) if hidden_layers_num > 0 else None

        self.output_layer = nn.Linear(hidden_dim, output_dim)

        self.softmax = nn.Softmax() if apply_softmax else None

    def forward(self, x):
        x = F.relu(self.input_layer(x))

        if self.hidden_layers:
            x = F.relu(self.hidden_layers(x))

        x = self.output_layer(x)

        if self.softmax:
            x = self.softmax(x)

        return x


all_models = {
    "LinearModel": LinearModel
}
