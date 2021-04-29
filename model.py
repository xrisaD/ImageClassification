import numpy as np
from utils import softmax, extreme_softmax

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
        w1_x = self.X @ np.transpose(self.W1) # Nb*M
        print()
        print("W11_XXX    ",w1_x)
        print()
        self.Z = self.activation.function(w1_x)  # z: Nb * M
        self.Z = np.hstack((np.ones((self.Z.shape[0], 1)), self.Z))  # z: Nb * (M+1)
        w2_z = self.Z @ np.transpose(self.W2)  # Nb*K
        self.Y = softmax(w2_z)  # Nb*K
        print()
        print("-----------------------")
        print("11111st Y  ",self.Y)
        return self.Y

    # returns the loss
    def likelihood(self, T, Y):
        # Y: predictions
        # T: true
        frobenius_norm = np.sum(np.power(np.absolute(self.W1), 2)) + np.sum(np.power(np.absolute(self.W2), 2))
        return np.sum(T * np.log(Y)) - self.lambdapar * frobenius_norm

    def backward(self, T):
        # update W2
        w2_grad = self.w2_derivative(T)
        self.W2 = self.W2 + self.learning_rate * w2_grad
        print()
        print("W2       ", w2_grad)
        # update W1
        w1_grad = self.w1_derivative(T)
        self.W1 = self.W1 + self.learning_rate * w1_grad
        print()
        print("W1       ", w1_grad)

    def w1_derivative(self, T):
        # t_y = np.transpose(T - self.Y)
        # activation_result = self.activation.derivative(self.W1 @ np.transpose(self.X))
        # W2_copy = np.copy(self.W2)
        # W2_sm = np.delete(W2_copy,  0, 1) # 0 sto 2o
        #
        # one = ((np.transpose(W2_sm) @ t_y) * activation_result)
        # # print()
        # # print("WWWWWWWWWWWWWWWW2")
        # # print(self.W2)

        t_y = T - self.Y
        activation_result = self.activation.derivative(self.W1 @ np.transpose(self.X))
        W2_copy = np.copy(self.W2)
        W2_sm = np.delete(W2_copy, 0, 1)
        one = np.transpose(t_y @ W2_sm) * activation_result
        return one @ self.X - self.lambdapar * self.W1

    def w2_derivative(self, T):
        print()
        print("Computing W2 grad..")
        t_y = np.transpose(T - self.Y)
        t_y_z = t_y @ self.Z
        print()
        print("TTTTT   ", T)
        print("YYYYY   ",self.Y)
        print("zZZZZZZZZZZZzzz    ", self.Z)
        print()
        print("praksiii ", t_y)
        print("MEGALOOOO ", t_y_z)
        #print("praksiii ", t_y_z) # EDW MEGALOOOO
        return t_y_z - self.lambdapar * self.W2

    def set_W1(self, W1):
        self.W1 = W1

    def step(self, X, T):
        print("")
        print("STEEEP")
        self.forward(X)
        self.backward(T)
        return self.likelihood(T, self.Y)
