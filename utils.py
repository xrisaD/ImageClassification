import numpy as np


def extreme_softmax(x):
    """ Compute the softmax function for each row of the input x.
    Arguments:
    x: N x K dimensional numpy matrix.
    Return:
    x: A N dimensional vector
    """
    tmp = np.max(x, axis=1)  # max per row
    x -= tmp.reshape((x.shape[0], 1))
    x = np.exp(x)

    tmp = np.sum(x, axis=1)
    x /= tmp.reshape((x.shape[0], 1))

    return x


def h_1(a):
    return np.log(1 + np.exp(a))


def h_2(a):
    return (np.exp(a) - np.exp(-a)) / (np.exp(a) + np.exp(-a))


def h_3(a):
    return np.cos(a)

def E(t, y, learning_rate):
    return t*np.log(y) #add learning rate


def softmax(x):
    """ Compute the softmax function for each row of the input x.
    Arguments:
    x: N x K dimensional numpy matrix.
    Return:
    x: A N dimensional vector
    """
    return np.exp(x) / np.transpose(np.tile(np.sum(np.exp(x), axis=1), (x.shape[1], 1)))
