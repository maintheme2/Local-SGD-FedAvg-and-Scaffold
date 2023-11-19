from model import all_models


class Client:
    def __init__(self):
        self.local_model = None
        self.local_weights = None
        self.local_dataset = None

    def init_local_model(self, model_name, model_params):
        self.local_model = all_models[model_name]
