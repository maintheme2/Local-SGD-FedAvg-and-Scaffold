class Server:
    def __init__(self, model_name, model_params, clients_num):
        self.model_name = model_name
        self.model_params = model_params
        self.clients_num = clients_num
        self.client = self.create_clients(model_name)

    def create_clients(self, model_name):
        return []
