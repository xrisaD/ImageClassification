import numpy as np

from activations import H3Activation
from datasets import Dataset
from metrics import f1, accuracy
from model import Model
import numpy.testing as nt

def train(n_epochs=3):
    np.random.seed(1)

    dataset = Dataset("mnist/test")  # TODO: change to train
    num_feats = dataset.get_num_feats()
    model_output_size = dataset.get_model_output_size()
    model = Model(num_feats, model_output_size, 100, H3Activation)

    print("Start...")
    w1 = model.W1
    w2 = model.W2

    for epoch in range(n_epochs):
        print(epoch)
        for X, Y_true in iter(dataset):
            model.step(X, Y_true)

    w1_a = model.W1
    w2_a = model.W2
    #nt.assert_array_almost_equal(w1, w1_a)
    #nt.assert_array_almost_equal(w2, w2_a)
    # loss test
    dataset = Dataset("mnist/test")
    X, T = dataset.get_all_examples()
    Y_pred = model.forward(X)
    loss = model.loss(T, Y_pred, 0.01)
    print(loss)

    print(f1(T.tolist(), Y_pred.tolist()))
    print(accuracy(T.tolist(), Y_pred.tolist()))


train(2)
