import sys

from datasets import Dataset, load_MNISTdata, load_CIFARdata
from sklearn.model_selection import train_test_split

from functions import run


def mnist():
    print("Load mnist data...")
    # load dataset
    X_train, Y_train = load_MNISTdata("mnist/train")
    X_test, Y_test = load_MNISTdata("mnist/test")

    # split train to train and dev
    X_train, X_dev, Y_train, Y_dev = train_test_split(X_train, Y_train, test_size=0.2, random_state=42, shuffle=True)

    train_dataset = Dataset(X_train, Y_train)
    dev_dataset = Dataset(X_dev, Y_dev)
    test_dataset = Dataset(X_test, Y_test)

    run(train_dataset, dev_dataset, test_dataset)


def cifar():
    print("Load cifar data...")

    X_train, X_test, Y_train, Y_test = load_CIFARdata()

    # split train to train and dev
    X_train, X_dev, Y_train, Y_dev = train_test_split(X_train, Y_train, test_size=0.2, random_state=42, shuffle=True)

    train_dataset = Dataset(X_train, Y_train)
    dev_dataset = Dataset(X_dev, Y_dev)
    test_dataset = Dataset(X_test, Y_test)

    run(train_dataset, dev_dataset, test_dataset)


if __name__ == "__main__":
    dataset_name = sys.argv[1]

    if dataset_name == 'mnist':
        mnist()
    elif dataset_name == 'cifar':
        cifar()
