import numpy as np


def xavier(fout, fin):
    return (np.random.rand(fout, fin) * 2 - 1) * np.sqrt(1 / fin)


def he(fout, fin):
    return (np.random.rand(fout, fin) * 2 - 1) * np.sqrt(2 / fin)


def glorot(fout, fin):
    return (np.random.rand(fout, fin) * 2 - 1) * np.sqrt(6 / (fin + fout))
