import numpy as np
from utils import softmax

from activations import H1Activation
from w_init import xavier


class Model:
    def __init__(self, D, K, learning_rate, lambdapar=0, M=100, activation=H1Activation, init=xavier, check=False,
                 W1=None, W2=None):
        # initialize weights
        # W1: M*(D+1)
        # W2: K*(M+1)
        self.lambdapar = lambdapar
        self.learning_rate = learning_rate

        self.W1 = init(M, D + 1)
        self.W2 = init(K, M + 1)

        if check:
            self.W1 = W1
            self.W2 = W2
        self.activation = activation

    # returns the predictions
    def forward(self, X):
        # X: Nb*D
        self.X = np.hstack((np.ones((X.shape[0], 1)), X))  # X: Nb*(D+1)
        w1_x = self.X @ np.transpose(self.W1)  # Nb*M
        self.Z = self.activation.function(w1_x)  # z: Nb * M
        self.Z = np.hstack((np.ones((self.Z.shape[0], 1)), self.Z))  # z: Nb * (M+1)
        w2_z = self.Z @ np.transpose(self.W2)  # Nb*K
        self.Y = softmax(w2_z)  # Nb*K
        return self.Y

    # returns the loss
    def likelihood(self, T, Y):
        # Y: predictions
        # T: true
        frobenius_norm = (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)))
        regularization_term = (self.lambdapar / 2) * frobenius_norm
        return np.sum(T * np.log(Y)) - regularization_term

    def backward(self, T):
        # compute grads
        w2_grad = self.w2_gradient(T)
        w1_grad = self.w1_gradient(T)
        # updates parameters
        self.W2 = self.W2 + self.learning_rate * w2_grad
        self.W1 = self.W1 + self.learning_rate * w1_grad

    def w1_gradient(self, T):
        t_y = T - self.Y
        activation_result = self.activation.derivative(self.W1 @ np.transpose(self.X))
        W2_copy = np.copy(self.W2)
        W2_sm = np.delete(W2_copy, 0, 1)
        return (np.transpose(t_y @ W2_sm) * activation_result) @ self.X - self.lambdapar * self.W1

    def w2_gradient(self, T):
        t_y = np.transpose(T - self.Y)
        t_y_z = t_y @ self.Z
        return t_y_z - self.lambdapar * self.W2

    def set_W1(self, W1):
        self.W1 = W1

    def set_W2(self, W2):
        self.W2 = W2

    def step(self, X, T):
        self.forward(X)
        self.backward(T)
        return self.likelihood(T, self.Y)
