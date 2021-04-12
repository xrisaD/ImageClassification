import numpy as np
from numpy import linalg as LA

def loss(Y, T, lambda1, lambda2, model):
    # Y: predictions
    # T: true
    np.sum(T*Y) - lambda1 * LA.norm(model.W1) - lambda2 * LA.norm(model.W2)