import numpy as np
from utils import softmax

from activations import H1Activation


class Model:
    def __init__(self, D, K, M=100, activation=H1Activation, check=False, W1=None):
        # initialize weights
        # W1: M*(D+1)
        # W2: K*(M+1)

        self.W1 = np.random.rand(M, D + 1)
        self.W2 = np.random.rand(K, M + 1)
        if check:
            self.W1 = W1

        self.activation = activation

    # returns the predictions
    def forward(self, X):
        # X: Nb*D
        self.X = np.hstack((np.ones((X.shape[0], 1)), X))  # X: Nb*(D+1)
        w1_x = self.X @ np.transpose(self.W1)  # M
        self.Z = self.activation.function(w1_x)  # z: Nb * M
        self.Z = np.hstack((np.ones((self.Z.shape[0], 1)), self.Z))  # z: Nb * (M+1)
        w2_z = self.Z @ np.transpose(self.W2)  # Nb*K
        self.Y = softmax(w2_z)  # Nb*K
        return self.Y

    # returns the loss
    def loss(self, T, Y, lambdapar):
        # Y: predictions
        # T: true
        frobenius_norm = np.sum(np.power(np.absolute(self.W1), 2)) + np.sum(np.power(np.absolute(self.W2), 2))
        return np.sum(T * np.log(Y)) - lambdapar * frobenius_norm

    def backward(self, T, learning_rate):

        t_y = np.transpose(T - self.Y)
        # update W2
        w2_grad = self.w2_derivative(t_y)
        self.W2 = self.W2 + learning_rate * w2_grad

        # update W1
        w1_grad = self.w1_derivative(t_y)
        self.W1 = self.W1 + learning_rate * w1_grad

    def w1_derivative(self, t_y):
        activation_result = self.activation.derivative(self.W1 @ np.transpose(self.X))
        W2_copy = np.copy(self.W2)
        W2_sm = np.delete(W2_copy, 0, 1)
        return ((np.transpose(W2_sm) @ t_y) * activation_result) @ self.X + self.lambdapar * self.W1

    def w2_derivative(self, t_y):
        return t_y @ self.Z - self.lambdapar * self.W2

    def step(self, X, T):
        self.lambdapar = 0
        self.forward(X)
        self.backward(T, 0.0001)