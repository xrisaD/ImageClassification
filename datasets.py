import numpy as np


def load_MNISTdata(set):
    X = []
    Y = []
    for i in range(0, 10):
        fileName = set + str(i) + ".txt"
        with open(fileName) as f:
            lines = f.readlines()

        for line in lines:
            X.append([int(i) for i in line.split()])
            Y.append(i)
    return np.array(X), np.array(Y)


# X_test = load_MNISTdata('mnist/test')

def random_mini_batches(self, X):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    num_batch = X.shape[0] // self.batch_size

    extras = X.shape[0] % self.batch_size

    fullindices = indices[: -extras]
    extraindices = indices[-extras:]

    finalindices = []
    for i in range(0, len(fullindices), self.batch_size):
        finalindices.append(fullindices[i: i + self.batch_size])

        finalindices.append(extraindices)

    return finalindices


class Dataset():
    def __init__(self, set, minibatches_size=100):
        self.batches = []
        self.X, self.Y = load_MNISTdata(set)
        self.minibatches_size = minibatches_size
        self.batches_creation()

    def batches_creation(self):
        # create batches
        indices = np.arange(self.X.shape[0])
        np.random.shuffle(indices)
        for i in range(0, len(indices), self.minibatches_size):
            self.batches.append(np.take(self.X, indices[i:min(i + self.minibatches_size, len(indices))], axis=0))  # a new batch
        self.batches = np.array(self.batches)

    def __iter__(self):
        self.n = -1
        return self

    def __next__(self):
        if self.n <= self.batches.shape[0]:
            self.n = self.n + 1
            return self.batches[self.n]
        else:
            raise StopIteration
