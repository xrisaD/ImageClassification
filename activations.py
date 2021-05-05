import numpy as np


class H1Activation:
    @staticmethod
    def function(x):
        return np.log(1 + np.exp(x))

    @staticmethod
    def derivative(x):
        return np.exp(x)/(1+np.exp(x))


class H2Activation:
    @staticmethod
    def function(a):
        return (np.exp(a) - np.exp(-a)) / (np.exp(a) + np.exp(-a))

    @staticmethod
    def derivative(a):
        return 1 - (H2Activation.function(a) ** 2)


class H3Activation:
    @staticmethod
    def function(a):
        return np.cos(a)

    @staticmethod
    def derivative(a):
        return - np.sin(a)
