import numpy as np


def load_MNISTdata(set):
    X = []
    Y = []

    for i in range(0, 10):
        fileName = set + str(i) + ".txt"
        with open(fileName) as f:
            lines = f.readlines()

        for line in lines:
            X.append([int(i) / 255 for i in line.split()])  # normalization
            Y.append(i)
    return X, create_one_hot(Y)


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def create_one_hot(Y):
    Y = np.array(Y).reshape(-1)
    one_hots = np.eye(10)[Y]
    return one_hots


def create_one_hot_batches(Y):
    one_hots = np.eye(10)[Y]
    return one_hots


def load_CIFARdata(path="cifar"):

    # load train
    X_train = []
    Y_train = []
    for i in range(1, 6):
        train_data = unpickle(path + "/data_batch_" + str(i))
        X_train.append(train_data[b'data'])
        Y_train.append(train_data[b'labels'])
    X_train = np.concatenate(X_train, axis=0)
    X_train = X_train/255  # normalization

    Y_train = np.concatenate(Y_train, axis=0)

    # load test
    test_data = unpickle(path + "/test_batch")
    X_test = test_data[b'data']
    Y_test = test_data[b'labels']
    X_test = X_test /255  # normalization

    return X_train, X_test, create_one_hot(Y_train), create_one_hot(Y_test)


class Dataset:
    def __init__(self, X, Y, batch_size=200):
        self.Xbatches = []
        self.Ybatches = []
        self.batch_size = batch_size

        if not isinstance(X, np.ndarray):
            self.X = np.array(X)
        else:
            self.X = X
        if not isinstance(Y, np.ndarray):
            self.Y = np.array(Y)
        else:
            self.Y = Y
        self.num_feats = self.X.shape[1]
        self.model_output_size = self.Y.shape[1]
        self.batches_creation()

    def batches_creation(self):
        # create batches
        indices = np.arange(self.X.shape[0])
        np.random.shuffle(indices)
        for i in range(0, len(indices), self.batch_size):
            self.Xbatches.append(
                np.take(self.X, indices[i:min(i + self.batch_size, len(indices))], axis=0))  # a new batch
            self.Ybatches.append(np.take(self.Y, indices[i:min(i + self.batch_size, len(indices))], axis=0))
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
