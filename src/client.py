from src.model import all_models
import torch.utils.data


class Client:
    def __init__(self, model_name, model_params, batch_size):
        self.model_name = model_name
        self.model_params = model_params

        self.model = None

        self.train_dataset = None
        self.train_dataloader = None
        self.batch_size = batch_size

        self.control_variate = 0
        self.control_variate_delta = 0

    def prepare(self, dataset):
        self.init_model()
        self.init_dataset(dataset)

    def init_model(self):
        self.model = all_models[self.model_name](**self.model_params)

    def init_dataset(self, dataset):
        self.train_dataset = dataset
        self.train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def local_update(self, epochs_num):
        self.init_model() # ???
        for i in range(epochs_num):
            for images, labels in self.df:
                # TODO: calculate gradients and update weights.
                ...
            # TODO: update control variate
            self.control_variate = ...