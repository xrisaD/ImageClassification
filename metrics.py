from sklearn.metrics import f1_score, accuracy_score

import numpy as np


def f1(y_true, y_pred, average="micro"):
    y_true = np.argmax(y_true, axis=1).tolist()
    y_pred = np.argmax(y_pred, axis=1).tolist()
    print(y_pred)
    return f1_score(y_true, y_pred, average=average)


def accuracy(y_true, y_pred):
    y_true = np.argmax(y_true, axis=1).tolist()
    y_pred = np.argmax(y_pred, axis=1).tolist()

    return accuracy_score(y_true, y_pred)
