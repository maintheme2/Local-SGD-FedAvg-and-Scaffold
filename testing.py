from FedAvg.fed_avg import FederatedAveraging

if __name__ == "__main__":
    fa = FederatedAveraging(model_params={
        'input_dim': 784,
        'hidden_dim': 200,
        'output_dim': 10
    }, device="cpu")
    fa.prepare()

    fa.train()
