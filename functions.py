from itertools import product

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


def tune(train_dataset, dev_dataset, epochs=5):
    # add activations
    activations = [H1Activation, H2Activation, H3Activation]
    inits = [xavier, he, glorot]
    lrs = [0.0001, 0.00001]
    ls = [0.00001, 0.000001]
    Ms = [100, 200, 300]

    best_acc = 0
    X, T = dev_dataset.get_all_examples()

    for (lr, l, M, activation, init) in product(lrs, ls, Ms, activations, inits):
        print()
        print("----------------------------------------------------")
        print(lr, l, M, str(activation), str(init))
        model = train(train_dataset, epochs, lr, l, M, activation, init)
        Y = model.forward(X)
        acc = accuracy(T, Y)
        print(" Accuracy is: ", acc)
        print("----------------------------------------------------")
        print()
        if best_acc < acc:
            best_acc = acc
            best_lr = lr
            best_l = l
            best_M = M
            best_activation = activation
            best_init = init
    print()
    print("----------------------------------------------------")
    print("Best hyperparameters: ", best_acc, best_lr, best_l, best_M, best_activation, best_init)
    print("----------------------------------------------------")
    print()
    return best_lr, best_l, best_M, best_activation, best_init


def run(train_dataset, dev_dataset, test_dataset):
    epochs = 60
    print("Start tuning...")
    best_lr, best_l, best_M, best_activation, best_init = tune(train_dataset, dev_dataset, epochs)

    print("Start training...")
    model = train(train_dataset, epochs, best_lr, best_l, best_M, best_activation, best_init)

    print("Test results: ")
    X, T = test_dataset.get_all_examples()
    Y_pred = model.forward(X)
    loss = model.likelihood(T, Y_pred)
    print(loss)
    print(accuracy(T, Y_pred))
