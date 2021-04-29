import numpy as np
import math

def xavier(fout, fin):
    limit = math.sqrt(1/fin)
    return np.random.uniform(-limit, limit, size=(fout, fin))
    #return (np.random.rand(fout, fin) * 2 - 1) * np.sqrt(1 / fin)


def he(fout, fin):
    #return (np.random.rand(fout, fin) * 2 - 1) * np.sqrt(2 / fin)
    limit = math.sqrt(2 / fin)
    return np.random.uniform(-limit, limit, size=(fout, fin))


def glorot(fout, fin):
    #return (np.random.rand(fout, fin) * 2 - 1) * np.sqrt(6 / (fin + fout))
    limit = math.sqrt(6 / (fin + fout))
    return np.random.uniform(-limit, limit, size=(fout, fin))
