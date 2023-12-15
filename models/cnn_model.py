import torch
from torch import nn
import torch.nn.functional as F

from models.base_model import BaseModel


class CNNModel(BaseModel):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 optimizer_func=None, hidden_layers_num=1,
                 apply_softmax=False):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.hidden_layers_num = hidden_layers_num
        self.apply_softmax = apply_softmax

        super().__init__(optimizer_func=optimizer_func)

    def initialize_weights(self):
        if self.hidden_layers_num < 0:
            print(f"Hidden layers num {self.hidden_layers_num} is less than 0. Resetting to 0...")
            self.hidden_layers_num = 0

        self.input_layer = nn.Conv2d(in_channels=1, out_channels=self.hidden_dim[0], kernel_size=5)

        self.hidden_layers = [
            nn.Conv2d(in_channels=self.hidden_dim[i + 1], out_channels=self.hidden_dim[i + 2], kernel_size=5)
            for i in range(self.hidden_layers)
        ] if self.hidden_layers_num > 0 else None

        self.output_layer = nn.Linear(512, self.output_dim)

        self.softmax = nn.Softmax() if self.apply_softmax else None

    def forward(self, x):
        x = F.max_pool2d(self.input_layer(x), kernel_size=2)

        if self.hidden_layers:
            for layer in self.hidden_layers:
                x = F.max_pool2d(layer(x), kernel_size=2)

        x = x.view(x.shape[0], -1)

        x = self.output_layer(x)

        if self.softmax:
            x = self.softmax(x)

        return x
