import torch
from torch import nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    def __init__(self, optimizer_func=None):
        super().__init__()

        self.initialize_weights()

        self.update_weights = optimizer_func
        self.control_variate = {
            key: torch.zeros(value.shape)
            for key, value in self.state_dict().items()
        }
        self.delta_control_variate = {
            key: torch.zeros(value.shape)
            for key, value in self.state_dict().items()
        }
        self.delta_weights = {
            key: torch.zeros(value.shape)
            for key, value in self.state_dict().items()
        }

    def initialize_weights(self):
        raise NotImplementedError("Child class should have 'initialize_weights' method")

    def get_weights(self):
        return self.state_dict(keep_vars=True)

    def aggregate_weights(self, weights, mode="mean"):
        new_weights = {}

        if mode == "mean":
            for key in weights[0].keys():
                new_weights[key] = torch.stack([weights[i][key] for i in range(len(weights))]).mean(0)
        else:
            raise NotImplementedError(f"Unknown aggregation mode {mode}!")

        return new_weights

    def forward(self, x):
        raise NotImplementedError("Forward method should be implemented in child classes")

    def train_step(self, images, labels):
        self.zero_grad()

        outputs = self.forward(images)
        loss = F.cross_entropy(outputs.cpu(), labels.cpu())

        loss.backward()
        self.update_weights(weights=self.get_weights(), c=self.control_variate)

        return loss.item()
