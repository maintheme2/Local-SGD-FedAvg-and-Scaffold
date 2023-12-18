import numpy as np
import torch
from tqdm import tqdm

from FedAvg.fed_avg import FederatedAveraging
from src.server import Server


class Scaffold(FederatedAveraging):
    def __init__(self, global_lr=1, **kwargs):
        super().__init__(**kwargs)

        self.global_lr = global_lr

    def prepare(self):
        self.server = Server(self.model_name, self.model_params,
                             super().update_weights, self.update_weights,
                             self.clients_num, self.threads_num,
                             self.global_lr, self.batch_size, self.dataset_name, self.device)
        self.server.prepare()

    def train(self, verbose=True):
        print("START TRAINING...") if verbose else None
        for round in range(self.rounds_num):
            print('-' * 20)
            print("Round:", round + 1)

            clients = np.random.choice(self.server.clients, int(self.clients_num * self.client_fraction), replace=False)
            print("Sending global weights to clients...") if verbose else None

            self.server.send_global_weights_to_clients(clients)

            print("Clients training...") if verbose else None

            self.round_step(clients, verbose)

            print("Testing server model...") if verbose else None

            self.server.test_step()

            print('-' * 20)

    def round_step(self, clients, verbose):
        clients_bar = tqdm(clients)

        for client in clients_bar:
            client.local_update(self.epochs_num)
            client.update_control_variate(self.server.global_model, self.epochs_num, self.lr)

        print("Updating global control variate...") if verbose else None
        self.server.update_control_variate(clients, self.client_fraction)

        print("Updating server model...") if verbose else None
        self.server.global_update(clients)

    def update_weights(self, weights, lr=0.01, c=None):
        with torch.no_grad():
            for key, value in weights.items():
                value -= lr * (value.grad - c[key] + self.server.global_model.control_variate[key])
