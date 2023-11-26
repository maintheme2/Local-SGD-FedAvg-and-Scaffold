import numpy as np

from FedAvg.fed_avg import FederatedAveraging


class Scaffold(FederatedAveraging):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.clients_control_variates = [0] * self.clients_num
        self.server_control_variate = 0

    def train(self, num_rounds, clients_num):
        for round in range(num_rounds):
            self.clients = np.random.choice(self.clients, clients_num, replace=False)
            self.send_global_weights_to_clients(self.clients)
            for client in self.clients:
                client.weights, self.clients_control_variates[client.id] = client.local_update(self.epochs_num)
