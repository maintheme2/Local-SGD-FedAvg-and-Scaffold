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

        self.optimizer = torch.optim.Adam(self.parameters())

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        x = F.relu(self.input_layer(x))

        if self.hidden_layers:
            x = F.relu(self.hidden_layers(x))

        x = self.output_layer(x)

        if self.softmax:
            x = self.softmax(x)

        return x

    def train_step(self, images, labels):
        self.zero_grad()

        outputs = self.forward(images)
        loss = F.cross_entropy(outputs.cpu(), labels.cpu())

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def get_weights(self):
        return self.state_dict()

    def aggregate_weights(self, weights, mode="mean"):
        new_weights = {}

        if mode == "mean":
            for key in weights[0].keys():
                new_weights[key] = torch.stack([weights[i][key] for i in range(len(weights))]).mean(0)
        else:
            raise NotImplementedError(f"Unknown aggregation mode {mode}!")

        self.load_state_dict(new_weights)


all_models = {
    "LinearModel": LinearModel
}
