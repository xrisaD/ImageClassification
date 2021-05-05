from activations import H3Activation, H1Activation, H2Activation
from datasets import Dataset
from model import Model
import random
import numpy as np
import numpy.testing as nt

from w_init import xavier


# model, W1, W2, X, T
def compute_numerical_gradient(model, W1, W2, X, T):
    print("Start computing..")

    model.set_W1(W1.copy())
    model.set_W2(W2.copy())
    e = 1e-6

    numerical_grad = np.zeros(W1.shape)
    for k in range(W1.shape[0]):
        for d in range(W1.shape[1]):
            # add epsilon to the w[k,d]
            W_tmp = np.copy(W1)
            W_tmp[k, d] += e
            model.set_W1(W_tmp)
            Y = model.forward(X)
            e_plus = model.likelihood(T, Y)

            # subtract epsilon to the w[k,d]
            W_tmp = np.copy(W1)
            W_tmp[k, d] -= e
            model.set_W1(W_tmp)
            Y = model.forward(X)
            e_minus = model.likelihood(T, Y)
            numerical_grad[k, d] = (e_plus - e_minus) / (2 * e)
    return numerical_grad


def grad_check():
    e = 1e-6

    # create fake data
    X_test = np.random.rand(3, 5)
    Y_test = [[0, 1, 0], [0, 1, 0], [0, 1, 0]]

    # get one batch
    dataset = Dataset(X_test, Y_test, 5)
    d = iter(dataset)
    X, T = next(d)
    num_feats = dataset.get_num_feats()
    model_output_size = dataset.get_model_output_size()

    # initialize W1 and W2
    W1 = xavier(100, num_feats + 1)
    W2 = xavier(model_output_size, 101)

    for activation in [H3Activation, H1Activation, H2Activation]:
        # initialize model and compute the gradient
        model = Model(num_feats, model_output_size, learning_rate=0.01, lambdapar=0.001, M=5, activation=activation,
                      check=True, W1=W1.copy(), W2=W2.copy())
        model.forward(X)
        grad = model.w1_gradient(T)

        numerical_grad = compute_numerical_gradient(model, W1, W2, X, T)

        print("The difference estimate for gradient of w is : ", np.max(np.abs(grad - numerical_grad)))

        if (np.abs(grad - numerical_grad) > e).any():
            print("Gradient check failed.")
        else:
            print("Success!")


grad_check()
