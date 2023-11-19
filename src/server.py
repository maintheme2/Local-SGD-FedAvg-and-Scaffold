import multiprocessing.dummy as mp

from src.client import Client
from src.model import all_models

import numpy as np
import torch
from torchvision.datasets import MNIST
from torch.utils.data import Subset

all_datasets = {
    "MNIST": MNIST
}


class Server:
    def __init__(self, model_name="LinearModel", model_params=None,
                 clients_num=10, threads_num=2,
                 batch_size=32, dataset_name="MNIST"):
        self.model_name = model_name
        self.model_params = model_params if model_params else {}

        self.clients_num = clients_num
        self.threads_num = threads_num
        self.process_pool = mp.Pool(threads_num)

        self.dataset_name = dataset_name
        self.batch_size = batch_size

        self.clients = []

        self.global_model = None

    def prepare(self):
        self.init_global_model()

        self.create_clients()
        self.prepare_clients()

    def init_global_model(self):
        self.global_model = all_models[self.model_name](**self.model_params)

    def create_clients(self):
        self.clients = self.process_pool.map(
            lambda _: Client(self.model_name, self.model_params, self.batch_size),
            range(self.clients_num)
        )

    def get_clients_parts(self):
        full_dataset = all_datasets[self.dataset_name]("./datasets", train=True, download=True)

        samples_per_client = len(full_dataset) // self.clients_num
        if samples_per_client < len(full_dataset.classes):  # TODO: add samples_per_client_threshold as parameter
            print(f"The dataset size = {len(full_dataset)} samples is too small for {self.clients_num} clients!")
            print("Some clients will have the same data!")

        # TODO: optimize stratified dataset splitting
        classes = np.unique(full_dataset.targets)
        idx_per_class = {
            cls: np.where(full_dataset.targets == cls)[0]
            for cls in classes
        }
        [np.random.shuffle(x) for x in idx_per_class.values()]

        clients_parts = [[] for _ in range(self.clients_num)]
        for cls in classes:
            indices = idx_per_class[cls]
            indices_per_client = len(indices) // self.clients_num

            if indices_per_client < 1:
                print(f"Warning: the number of samples for class {cls} is less"
                      f" than number of clients {self.clients_num}!")

            for i in range(0, len(indices) - indices_per_client, indices_per_client):
                indices_part = indices[i:min(i + indices_per_client, len(indices))]
                clients_parts[i // indices_per_client].extend(indices_part)

        clients_parts = [
            Subset(full_dataset, clients_parts[i])
            for i in range(self.clients_num)
        ]

        return clients_parts

    def prepare_clients(self):
        clients_parts = self.get_clients_parts()

        self.process_pool.starmap(
            lambda client, dataset: client.prepare(dataset),
            zip(self.clients, clients_parts)
        )
