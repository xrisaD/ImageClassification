from activations import H3Activation
from datasets import Dataset
from model import Model
import random
import numpy as np

rndstate = random.getstate()
e = 10e-6


def grad_check():
    dataset = Dataset("mnist/test")
    d = iter(dataset)
    X, T = next(d)

    model0 = model_init(dataset, 0)
    model0.step(X, T)
    model0W1 = model0.W1

    model1 = model_init(dataset, 1)
    model1.step(X, T)
    model1W1 = model1.W1

    model_minus_1 = model_init(dataset, -1)
    model_minus_1.step(X, T)
    model_minus_1W1 = model_minus_1.W1

    print("Compare..")
    x = model0W1 / ((model1W1 - model_minus_1W1) / 2)

    if not ((x > e).all()):
        print("Gradient check failed.")


def model_init(dataset, param):
    num_feats = dataset.get_num_feats()
    model_output_size = dataset.get_model_output_size()
    random.setstate(rndstate)
    W1 = np.random.rand(100, num_feats + 1) + np.full((100, num_feats + 1), param * e)
    model = Model(num_feats, model_output_size, 100, H3Activation, check=True, W1=W1)
    return model


grad_check()
