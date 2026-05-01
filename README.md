# Investigating-Function-Approximation-With-Neural-Networks
The Universal Approximation Theorem is examined in the context of function approximation. The theorem states that a neural network with a single hidden layer can approximate any continuous function arbitrarily well, given a sufficient number of neurons. A single-layer network with four neurons is implemented to approximate the function y=x^2 using different activation functions, including step, ReLU, and sigmoid. The network is initialized with specified weights and trained over 3000 epochs to minimize the mean squared error (MSE) between predicted and target outputs.

The investigation is further extended to additional functions to investigate the influence of model parameters on approximation performance.

For this study a series of scripts were designed to approximate and visualize the results. It is recommended to only adjust hyperparameters and see their effects on the approximation.

Each script produces two outputs: a GIF that visualizes the training process of the approximation, and a final plot showing the learned function along with the loss.
