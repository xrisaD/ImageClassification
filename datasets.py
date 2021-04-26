import numpy as np


def load_MNISTdata(set):
    min=0
    max=255
    X = []
    Y = []

    for i in range(0, 10):
        fileName = set + str(i) + ".txt"
        with open(fileName) as f:
            lines = f.readlines()

        for line in lines:
            X.append([(int(i)-min)/(max-min) for i in line.split()]) # normalization
            Y.append(i)
    #create one_hots
    Y = np.array(Y).reshape(-1)
    one_hots = np.eye(10)[Y]
    return X, one_hots.tolist()


class Dataset:
    def __init__(self, X, Y, minibatches_size=200):
        self.Xbatches = []
        self.Ybatches = []
        self.num_feats = len(X[0])
        self.model_output_size = len(Y[0])
        self.X = np.array(X)
        self.Y = np.array(Y)

        self.minibatches_size = minibatches_size
        self.batches_creation()

    def batches_creation(self):
        # create batches
        indices = np.arange(self.X.shape[0])
        np.random.shuffle(indices)
        for i in range(0, len(indices), self.minibatches_size):
            self.Xbatches.append(
                np.take(self.X, indices[i:min(i + self.minibatches_size, len(indices))], axis=0))  # a new batch
            self.Ybatches.append(np.take(self.Y, indices[i:min(i + self.minibatches_size, len(indices))], axis=0))
        self.Xbatches = np.array(self.Xbatches)
        self.Ybatches = np.array(self.Ybatches)

    def __iter__(self):
        self.n = -1
        return self

    def __next__(self):
        if self.n < self.Xbatches.shape[0] - 1:
            self.n = self.n + 1
            return self.Xbatches[self.n], self.Ybatches[self.n]
        else:
            raise StopIteration

    def get_model_output_size(self):
        return self.model_output_size

    def get_num_feats(self):
        return self.num_feats

    def get_all_examples(self):
        return self.X, self.Y
