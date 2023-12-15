from FedAvg.fed_avg import FederatedAveraging
from SCAFFOLD.scaffold import Scaffold

if __name__ == "__main__":
    # fa = FederatedAveraging(model_params={
    #     'input_dim': 784,
    #     'hidden_dim': 200,
    #     'output_dim': 10,
    #     'hidden_layers_num': 2
    # }, device="cpu")
    # fa.prepare()
    #
    # fa.train()

    sc = Scaffold(model_params={
        'input_dim': 784,
        'hidden_dim': 200,
        'output_dim': 10
    }, device="cpu")
    sc.prepare()

    sc.train()
