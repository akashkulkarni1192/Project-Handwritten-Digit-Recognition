import numpy as np
import pandas as pd


def error_rate(p, t):
    return np.mean(p != t)


def one_hot_encoding(Y):
    N = len(Y)
    K = len(set(Y))
    encodedY = np.zeros((N, K))
    for i in range(len(Y)):
        encodedY[i, Y[i]] = 1
    return encodedY


def get_test_data(filename):
    df = pd.read_csv(filename)
    data = df.as_matrix().astype(np.float32)
    X = data /255
    return X

def get_data(filename):
    df = pd.read_csv(filename)
    data = df.as_matrix().astype(np.float32)
    X = data[:, 1:]/255
    Y = data[:, 0].astype(np.int32)
    return X, Y



