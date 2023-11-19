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

    def prepare(self, dataset):
        self.init_model()
        self.init_dataset(dataset)

    def init_model(self):
        self.model = all_models[self.model_name](**self.model_params)

    def init_dataset(self, dataset):
        self.train_dataset = dataset
        self.train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
