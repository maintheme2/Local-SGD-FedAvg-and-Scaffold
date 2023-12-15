import multiprocessing as mp

import numpy as np
import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import Subset

from src.client import Client
from src.model import all_models

all_datasets = {
    "MNIST": MNIST
}


class Server:
    def __init__(self, model_name="LinearModel", model_params=None,
                 server_optimizer_func=None, client_optimizer_func=None,
                 clients_num=10, processes_num=2,
                 batch_size=32, dataset_name="MNIST", device="cpu"):
        self.model_name = model_name
        self.model_params = model_params if model_params else {}

        self.clients_num = clients_num
        self.processes_num = processes_num
        self.process_pool = mp.Pool(processes=self.processes_num)

        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.device = device

        self.test_dataset = None
        self.test_dataloader = None

        self.client_optimizer_func = client_optimizer_func
        self.server_optimizer_func = server_optimizer_func

        self.clients = []

        self.global_model = None

        self.logs = dict()
        self.logs['val_accuracy'] = []

    def prepare(self):
        self.init_global_model()
        self.prepare_test_dataset()

        self.create_clients()
        self.prepare_clients()

    def init_global_model(self):
        self.global_model = all_models[self.model_name](**self.model_params,
                                                        optimizer_func=self.server_optimizer_func).to(self.device)

    def prepare_test_dataset(self):
        self.test_dataset = all_datasets[self.dataset_name]("./datasets", train=False, download=True,
                                                            transform=transforms.ToTensor())
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def create_clients(self):
        self.clients = [
            Client(self.model_name, self.model_params, self.batch_size, self.client_optimizer_func)
            for _ in range(self.clients_num)
        ]

    def get_clients_parts(self):
        full_dataset = all_datasets[self.dataset_name]("./datasets", train=True, download=True,
                                                       transform=transforms.ToTensor())

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
        np.random.seed(478)
        [np.random.shuffle(x) for x in idx_per_class.values()]

        clients_parts = [[] for _ in range(self.clients_num)]
        for cls in classes:
            indices = idx_per_class[cls]
            indices_per_client = len(indices) // self.clients_num

            if indices_per_client < 1:
                print(f"Warning: the number of samples for class {cls} is less"
                      f" than number of clients {self.clients_num}!")

            for client_idx in range(self.clients_num):
                clients_parts[client_idx].extend(
                    indices[indices_per_client * client_idx:indices_per_client * (client_idx + 1)]
                )

        clients_parts = [
            Subset(full_dataset, clients_parts[i])
            for i in range(self.clients_num)
        ]

        return clients_parts

    def prepare_clients(self):
        clients_parts = self.get_clients_parts()

        for client, dataset in zip(self.clients, clients_parts):
            client.prepare(dataset)

    def global_update(self, clients):
        self.global_model.load_state_dict(
            self.aggregate_clients_weights(clients)
        )

    def aggregate_clients_weights(self, clients, mode='weights'):
        if mode == 'weights':
            client_weights = [client.model.get_weights() for client in clients]
        elif mode == 'delta_weights':
            client_weights = [client.model.delta_weights for client in clients]
        elif mode == 'delta_c':
            client_weights = [client.model.delta_control_variate for client in clients]
        else:
            client_weights = []

        weights = self.global_model.aggregate_weights(client_weights)

        return weights

    def send_global_weights_to_clients(self, clients):
        for client in clients:
            client.model.load_state_dict(self.global_model.state_dict())

    def update_control_variate(self, clients, client_fraction):
        self.global_model.delta_weights = self.aggregate_clients_weights(clients, mode='delta_weights')
        self.global_model.delta_control_variate = self.aggregate_clients_weights(clients, mode='delta_c')

        current_state_dict = self.global_model.get_weights()
        for key in current_state_dict.keys():
            current_state_dict[key].grad = -1 * self.global_model.delta_weights[key]
            self.global_model.control_variate[key] += client_fraction * self.global_model.delta_control_variate[key]

        self.global_model.update_weights(self.global_model.get_weights())

    def test_step(self):
        with torch.no_grad():
            correct = 0
            total = 0

            for images, labels in self.test_dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.global_model(images)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total

            self.logs['val_accuracy'].append(accuracy)

            print(f"Global Model Accuracy: {accuracy:.2f}%")
