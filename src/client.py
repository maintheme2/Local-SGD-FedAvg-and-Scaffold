from src.model import all_models
import torch.utils.data


class Client:
    def __init__(self, model_name, model_params, batch_size, optimizer_func, device="cpu"):
        self.model_name = model_name
        self.model_params = model_params

        self.model = None

        self.train_dataset = None
        self.train_dataloader = None
        self.batch_size = batch_size
        self.device = device

        self.optimizer_func = optimizer_func

        self.logs = dict()
        self.logs['rounds_num'] = 0
        self.logs['losses'] = []

    def prepare(self, dataset):
        self.init_model()
        self.init_dataset(dataset)

    def init_model(self):
        self.model = all_models[self.model_name](**self.model_params,
                                                 optimizer_func=self.optimizer_func).to(self.device)

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

    def update_control_variate(self, global_model, epochs_num, lr):
        global_weights = global_model.get_weights()
        local_weights = self.model.get_weights()

        for key, value in local_weights.items():
            c_new = (
                    (self.model.control_variate[key] - global_model.control_variate[key]) +
                    1 / (self.batch_size * epochs_num * lr) *
                    (global_weights[key] - value)
            )

            self.model.delta_weights[key] = value - global_weights[key]
            self.model.delta_control_variate[key] = c_new - self.model.control_variate[key]
            self.model.control_variate[key] = c_new
