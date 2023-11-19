from src.server import Server


class FederateAveraging:
    def __init__(self, clients_num=10, rounds_num=10, epochs_num=2, client_fraction=0.2,
                 dataset_name="MNIST", model_name="LinearModel", model_params=None,
                 batch_size=32, lr=0.001, loss="crossentropy",
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
                             self.clients_num, self.threads_num,
                             self.batch_size, self.dataset_name)
        self.server.prepare()


