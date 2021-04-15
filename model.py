import numpy as np
from utils import h_1, softmax

class Model:
    def __init__(self, K, M=100, activation=h_1):
        # initialize weights
        # W1: M*(D+1)
        # W2: K*(M+1)
        self.W1 = np.random.rand(M, D + 1)
        self.W2 = np.random.rand(K, M + 1)
        self.activation = activation

    def forward(self, X):
        # X: Nb*D
        X = np.hstack(np.ones((X.shape[0], 1)), X) # X: Nb*(D+1)
        w1_x = X @ np.transpose(self.W1) # M
        self.Z = self.activation.function(w1_x) # z: Nb * M
        self.Z = np.hstack(np.ones((self.Z.shape[0],1)), z) # z: Nb * (M+1)
        w2_z = self.Z @ np.transpose(self.W2) # Nb*K
        self.Y = softmax(w2_z) #Nb*K
        return self.Y

    def loss(self, T, lambdapar):
        # Y: predictions
        # T: true
        frobenius_norm = np.sum(np.power(np.absolute(self.W1), 2)) + np.sum(np.power(np.absolute(self.W2), 2))
        return np.sum(T * self.Y) - lambdapar * frobenius_norm

    def backward(self, T, X, learning_rate, lambdapar):
        # update W1
        t_y = np.transpose(T-self.Y)
        W2_sm = self.W2.delete(0, 1)
        h_d = self.activation.derivative(self.W1 @ np.transpose(X)) * X
        w1_derivative = (W2_sm @ t_y) * h_d + lambdapar * self.W1
        self.W1 = self.W1 + learning_rate * w1_derivative

        # update W2
        w2_dericative = t_y*self.Z - lambdapar * self.W2
        self.W2 = self.W2 + learning_rate * w2_dericative
