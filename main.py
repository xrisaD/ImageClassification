
from datasets import Dataset, load_MNISTdata
from sklearn.model_selection import train_test_split

from functions import train, tune
from metrics import accuracy


def main():
    print("Load data...")
    # load dataset
    X_train, Y_train = load_MNISTdata("mnist/train")
    X_test, Y_test= load_MNISTdata("mnist/test")

    # split test to test and dev
    X_dev, X_test, y_dev, y_test = train_test_split(X_test, Y_test, test_size=0.5, random_state=42, shuffle=True)

    train_dataset = Dataset(X_train, Y_train)
    dev_dataset = Dataset(X_dev, y_dev)
    test_dataset = Dataset(X_test, y_test)

    print("Start tuning...")
    best_lr, best_l, best_M, best_activation, best_init = tune(train_dataset, dev_dataset, 60)

    print("Start training...")

    model = train(train_dataset, 40, best_lr, best_l, best_M, best_activation, best_init)

    print("Test results: ")
    X, T = test_dataset.get_all_examples()
    Y_pred = model.forward(X)
    loss = model.likelihood(T, Y_pred)
    print(loss)
    print(accuracy(T, Y_pred))



if __name__ == "__main__":
    main()