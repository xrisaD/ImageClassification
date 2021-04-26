import numpy as np

from tqdm import tqdm

from activations import H3Activation, H1Activation, H2Activation
from w_init import xavier, he, glorot
from metrics import accuracy
from model import Model


def train(train_dataset, epochs, lr, l, M, activation, init):
    p_bar = tqdm(range(epochs))
    model = Model(train_dataset.get_num_feats(), train_dataset.get_model_output_size(), lr, l, M, activation, init)
    for epoch in p_bar:
        for X, T in iter(train_dataset):
            likelihood = model.step(X, T)
            p_bar.set_postfix_str("likelihood %.4f" % likelihood)
    return model


def tune(train_dataset, dev_dataset, epochs=40):
    # add activations
    activations = [H1Activation, H2Activation, H3Activation]
    inits = [xavier, he, glorot]
    lrs = [0.0001, 0.00001]
    ls = [0.00001, 0.000001]
    Ms = [100, 200, 300]

    best_acc = 0
    X, T = dev_dataset.get_all_examples()

    for lr in lrs:
        for l in ls:
            if lr > l:
                for M in Ms:
                    for activation in activations:
                        for init in inits:
                            model = train(train_dataset, epochs, lr, l, M, activation, init)
                            Y = model.forward(X)
                            acc = accuracy(T, Y)
                            print(lr, l, M, " Accuracy is: ", acc)
                            if best_acc < acc:
                                best_acc = acc
                                best_lr = lr
                                best_l = l
                                best_M = M
                                best_activation = activation
                                best_init = init

    print("Best hyperparameters: ", best_acc, best_lr, best_l, best_M, best_activation, best_init)

    return best_lr, best_l, best_M
