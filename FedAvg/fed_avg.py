import numpy as np
import torch
from tqdm import tqdm

from src.server import Server


class FederatedAveraging:
    def __init__(self, clients_num=100, rounds_num=80, epochs_num=1, client_fraction=0.2,
                 dataset_name="MNIST", model_name="LinearModel", model_params=None,
                 batch_size=10, lr=0.1, loss="crossentropy",
                 threads_num=2, device="cpu"):
        self.clients_num = clients_num
        self.rounds_num = rounds_num
        self.epochs_num = epochs_num
        self.client_fraction = client_fraction

        self.dataset_name = dataset_name
        self.model_name = model_name
        self.model_params = model_params if model_params else {}

        self.batch_size = batch_size
        self.lr = lr
        self.loss = loss

        self.threads_num = threads_num
        self.device = device

        self.server = None

    def prepare(self):
        self.server = Server(self.model_name, self.model_params,
                             self.update_weights, self.update_weights,
                             self.clients_num, self.threads_num,
                             self.batch_size, self.dataset_name, self.device)
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

        print("Aggregating clients weights...") if verbose else None
        self.server.global_update(clients)

    def update_weights(self, weights, c=None):
        with torch.no_grad():
            for key, value in weights.items():
                value -= self.lr * value.grad
