import numpy as np
import pickle


def get_data():
    pickle_in = open("X_train.pickle", "rb")
    train_data = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open("y_train.pickle", "rb")
    train_labels = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open("X_test.pickle", "rb")
    test_data = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open("y_test.pickle", "rb")
    test_labels = pickle.load(pickle_in)
    pickle_in.close()

    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)
    return (train_data, train_labels), (test_data, test_labels)
