import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, List

from FedAvg.fed_avg import FederatedAveraging


class Scaffold(FederatedAveraging):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.clients_control_variates = Dict[List[torch.Tensor]] = {}
        self.delta_control_variates = []

        self.server_control_variate = 0

    # TODO: control variate updates
    def train(self, verbose=True):
        print("START TRAINING...") if verbose else None
        for round in range(self.num_rounds):
            print('-' * 20)
            print("Round:", round + 1)

            clients = np.random.choice(self.server.clients, int(self.clients_num * self.client_fraction), replace=False)
            if round == 0:
                # initialize control variates_i (for each client)
                for client in clients:
                    self.clients_control_variates[client] = []            
            print("Sending global weights to clients...") if verbose else None
            
            self.server.send_global_weights_to_clients(clients)
            
            print("Clients training...") if verbose else None
            
            self.round_step(clients, verbose)

            print("Validating server model...") if verbose else None
            
            self.server.validation_step()

            print('-' * 20)
        
         
    def round_step(self, clients, verbose):
        clients_bar = tqdm(clients)

        for client in clients_bar:
            client.local_update(self.epochs_num)
            # TODO: control variate updates

        print("Aggregating clients weights...") if verbose else None
        self.server.aggregate_clients_weights(clients)

