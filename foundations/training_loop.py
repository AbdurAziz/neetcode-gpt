import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))

        #Training loop that DRIVES LEARNING

        #FOUR STEPS:
        #1. Forward Pass Y = Xw + b
        #2. Loss Computation 
        #3. Gradient Calculation
        #4. Weight Update

        n = X.shape[0]
        w = np.zeros(X.shape[1])
        b = 0.0

        for i in range(epochs):
            #forward pass
            yHat = X @ w + b
            error = yHat - y

            #Find the gradients of MSE Loss
            dw = (2.0/n) * (X.T @ error)
            db = (2.0/n) * np.sum(error)

            w = w - lr * dw
            b = b - lr * db

        return (np.round(w, 5), round(float(b), 5))


        
