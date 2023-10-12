## Federated learning

In this paper, the authors propose a technique that will help users of sophisticated electronic devices diminish the risks of storing their private and sensitive data locally.
The proposed technique aims to address the critical issues of data security and privacy on electronic devices, which are essential in today's digital world. By reducing the risks associated with storing private and sensitive data, it can contribute to a safer and more secure digital environment for users.
The main idea of this technique is to provide a shared model that will learn using the training data from the distributed sources (e.g. mobile phones, network sensors). This decentralized approach is called Federated Learning. 

## SCAFFOLD: Stochastic Controlled Averaging for Federated Learning

In another paper, the authors provide an algorithm designed to improve the performance of federated learning - SCAFFOLD. It is noted that federated averaging (the algorithm of choice for federated learning) may suffer from "client-drift" when dealing with non-identically distributed data, leading to unstable and slow convergence. The author’s main goal is to correct this drift. 
They follow this idea: 
calculate the difference between the server model's update direction and each client's update direction, which gives an estimate of the "client-drift." This drift represents how different the updates from individual clients are from the global update. Then they use this estimated client drift to correct the local updates on each client. By making these corrections, SCAFFOLD aims to align the local updates with the global update direction.

## The basis of these approaches were:
The described techniques based mainly on the stochastic gradient descent (SGD) and it's variations. For example federated averaging algorithm combines localSGD on each client.
