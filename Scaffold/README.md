# SCAFFOLD Algorithm

Stochastic Controlled Averaging for Federated Learning

## Key Features

1. **Distributed learning**:
    - Data is distributed across multiple clients
    - Compute is distributed across multiple workers
2. **Robustness to client drift**:
    - Converges for arbitrarily heterogeneous data
3. **Speed of convergence**:
    - At least as fast as SGD
    - Faster than FedAvg
4. **Takes advantage of similarity between clients**
5. **Data privacy**

## Why *SCAFFOLD* can work well?

1. **Controlled Averaging**

The SCAFFOLD algorithm introduces a controlled averaging mechanism that allows the clients to
contribute their local model updates to the global model in a controlled manner. This controlled averaging helps in
mitigating the effects of heterogeneity among the clients and ensures that the global model benefits from the
contributions of all clients, even those with limited computational resources or noisy data.

2. **Adaptive Learning Rate**

The algorithm incorporates an adaptive learning rate scheme that dynamically adjusts
the learning rate based on the local model updates of the clients. This adaptive learning rate helps in achieving
faster convergence and better performance by adapting to the characteristics of the clients' data distributions.

3. **Communication Efficiency**

The SCAFFOLD algorithm reduces the amount of communication required in federated learning
by allowing the clients to communicate their local model updates selectively. Instead of communicating every local
update, the clients use a control mechanism to decide when to communicate their updates to the server. This selective
communication reduces the communication overhead and improves the overall efficiency of the federated learning
process.

4. **Theoretical Analysis**

The SCAFFOLD algorithm is supported by theoretical analysis that provides insights into its
convergence properties and performance guarantees. The paper presents theoretical results that demonstrate the
convergence of the SCAFFOLD algorithm under certain assumptions and provide bounds on the convergence rate.

## Improvements comparing to basic versions

1. **Faster than SGD and supports client sampling**: previous works propose variance reduction to deal
   with client heterogeneity, but they are slower than SGD and do not support client sampling.
2. **Proximal point update**: in contrast to some other works, where a fixed number of (stochastic) gradient steps are
   used to update the local model, the SCAFFOLD algorithm uses a proximal point update that allows the client to take
   multiple gradient steps to update its local model.