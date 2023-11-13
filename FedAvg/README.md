# *FederatedAveraging*(FedAvg) Algorithm

## Key features

1. **Distributed learning**:
    - Data is distributed across multiple clients
    - Compute is distributed across multiple workers
2. **Data privacy**:
    - Data is not shared between clients
    - Data is not uploaded to a central server
3. **Data heterogeneity**:
    - Data is not identically distributed across clients
    - Data is not identically distributed across time
4. **Communication efficiency**:
    - Communication is asynchronous
    - Communication is sparse
    - Communication is compressed

## Why *FedAvg* can work well?

1. **Aggregation of Local Models**

In the FedAvg algorithm, each client trains a local model
using its own data and then sends the model updates to a central server.
The server aggregates these updates by taking a weighted average of the models.
This aggregation process helps to combine the knowledge learned by individual clients,
resulting in a more accurate and robust global model.

2. **Mitigating Heterogeneity**

In federated learning, the data distribution across clients can be highly heterogeneous,
meaning that each client may have different types of data or data from different domains.
The FedAvg algorithm is designed to handle this heterogeneity by allowing each client to perform multiple local updates
before sending the model updates to the server.
This way, clients with more data or more complex data can contribute more to the global model, helping to mitigate the
impact of heterogeneity.

3. **Privacy Preservation**

Federated learning aims to train a global model without accessing or sharing the raw data from individual clients.
The FedAvg algorithm achieves this by only exchanging model updates between the clients and the server.
This privacy-preserving nature of the algorithm makes it suitable for scenarios where data privacy is a concern,
such as in healthcare or finance.

4. **Communication Efficiency**

In federated learning, communication between clients and the server can be a bottleneck due to limited bandwidth or high
latency. The FedAvg algorithm addresses this challenge by reducing the amount of communication required. Instead of
sending the entire model, clients only need to send the model updates, which are typically much smaller. This
communication efficiency allows federated learning to scale to a large number of clients.

5. **Convergence Guarantees**

The FedAvg algorithm has been shown to converge to a global optimum under certain conditions. Theoretical analyses and
empirical studies have demonstrated that FedAvg can achieve good convergence rates and performance in various settings,
including non-IID (non-independent and identically distributed) data.

## Improvements comparing to basic versions

1. **Unbalanced and non-IID datasets**: Previous works only consider the cluster / data center setting
   (at most 16 workers, wall-clock time based on fast networks),
   and do not consider datasets that are unbalanced and non-IID, properties that are essential to the federated learning
   setting

2. **Empirical evaluation**: Some other works emphasize the importance of privacy, and address communication costs by
   only sharing a subset of the parameters during each round of communication. However, these works do not provide
   enough empirical evidence to support their claims.

3. **Communication efficiency**: The communication cost of the algorithm is proportional to the number of parameters
   being transferred, which is independent of the number of workers. This is in contrast to previous works that require
   communication proportional to the number of workers.