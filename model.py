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

    def forward(self, x):
        # x: Nb*D
        x = np.hstack(np.ones((x.shape[0], 1)), x) # x: Nb*(D+1)
        w1_x = x @ np.transpose(self.W1) # M
        z = self.activation(w1_x) # z: Nb * M
        z = np.hstack(np.ones((z.shape[0],1)), z) # z: Nb * (M+1)
        w2_z = z @ np.transpose(self.W2) # Nb*K
        y = softmax(w2_z) #Nb*K
        return y
    def backward(self):
        return 0;