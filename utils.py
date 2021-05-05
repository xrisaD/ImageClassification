import numpy as np


def softmax(x):
    m = np.max(x, axis=1, keepdims=True)  # max per row
    p = np.exp(x - m)
    return (p / np.sum(p, axis=1, keepdims=True))
