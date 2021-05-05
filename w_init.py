import numpy as np
import math

def xavier(fout, fin):
    limit = math.sqrt(1/fin)
    return np.random.uniform(-limit, limit, size=(fout, fin))


def he(fout, fin):
    limit = math.sqrt(2 / fin)
    return np.random.uniform(-limit, limit, size=(fout, fin))


def glorot(fout, fin):
    limit = math.sqrt(6 / (fin + fout))
    return np.random.uniform(-limit, limit, size=(fout, fin))
