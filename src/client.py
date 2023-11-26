from src.model import all_models
import torch.utils.data


class Client:
    def __init__(self, model_name, model_params, batch_size, device="cpu"):
        self.model_name = model_name
        self.model_params = model_params

        self.model = None

        self.train_dataset = None
        self.train_dataloader = None
        self.batch_size = batch_size
        self.device = device

        self.control_variate = 0
        self.control_variate_delta = 0

        self.logs = dict()
        self.logs['rounds_num'] = 0
        self.logs['losses'] = []

    def prepare(self, dataset):
        self.init_model()
        self.init_dataset(dataset)

    def init_model(self):
        self.model = all_models[self.model_name](**self.model_params).to(self.device)

    def init_dataset(self, dataset):
        self.train_dataset = dataset
        self.train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def local_update(self, epochs_num):
        total_loss = 0
        for epoch in range(epochs_num):
            epoch_loss = 0
            for images, labels in self.train_dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                epoch_loss += self.model.train_step(images, labels)

            total_loss += epoch_loss / len(self.train_dataloader)

        self.logs['losses'].append(total_loss / epochs_num)
        self.logs['rounds_num'] += 1

        return self.model.get_weights()
