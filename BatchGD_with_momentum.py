import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import ann_function as myneural

def one_hot_encoding(Y):
    N = len(Y)
    K = len(set(Y))
    encodedY = np.zeros((N, K))
    for i in range(len(Y)):
        encodedY[i, Y[i]] = 1
    return encodedY


def normalize_pca(X):
    mu = X.mean(axis=0)
    X = X - mu  # center the data
    pca = PCA()
    Z = pca.fit_transform(X)
    return Z


def getdata():
    dataframe = pd.read_csv('train.csv')
    data = dataframe.as_matrix().astype(np.float32)
    np.random.shuffle(data)
    X = data[:, 1:]
    Z = normalize_pca(X)
    Y = data[:, 0].astype(np.int32)
    return Z, Y

def normalize(X):
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mu) / std
    return X


if __name__ == '__main__':
    X, Y = getdata()

    X = normalize(X)

    Xtrain = X[:-1000, ]
    Ytrain = Y[:-1000]
    Xtest = X[-1000:, ]
    Ytest = Y[-1000:]

    Ttrain = one_hot_encoding(Ytrain)
    Ttest = one_hot_encoding(Ytest)
    # initializing dimensions
    N, D = Xtrain.shape
    K = len(set(Y))
    M = 300
    K = 10

    # initializing initial weights
    W1 = np.random.randn(D, M) / 28
    b1 = np.zeros(M)
    W2 = np.random.randn(M, K) / np.sqrt(M)
    b2 = np.zeros(K)

    # initializing hyper parameters
    lr = 0.00004
    reg = 0.01
    batch_size = 500

    # 1. Batch Gradient Descent
    n_batches = int(N / batch_size)
    cost_list_batch = []
    iter = 20
    for i in range(int(iter)):
        tempX, tempY = shuffle(Xtrain, Ttrain)
        for j in range(n_batches):
            batchX = tempX[j * batch_size: (j * batch_size + batch_size), :]
            batchT = tempY[j * batch_size: (j * batch_size + batch_size), :]

            Y, Z = myneural.forward(batchX, W1, b1, W2, b2)

            W2 -= lr * (myneural.derivative_w2(Z, batchT, Y) + reg * W2)
            b2 -= lr * (myneural.derivative_b2(batchT, Y) + reg * b2)
            W1 -= lr * (myneural.derivative_w1(batchX, Z, batchT, Y, W2) + reg * W1)
            b1 -= lr * (myneural.derivative_b1(Z, batchT, Y, W2) + reg * b1)

            Y, __ = myneural.forward(Xtest, W1, b1, W2, b2)
            singlecost = myneural.cost(Y, Ttest)
            cost_list_batch.append(singlecost)
            print('Iter : {0} Batch : {1} Cost : {2}'.format(i, j, myneural.cost(Y, Ttest)))
            # if (j % (10) == 0):
            #     print('Accuracy rate : {0}'.format(myneural.error_rate(Y, Ytest)))

    x1 = np.linspace(0, 1, len(cost_list_batch))
    plt.plot(x1, cost_list_batch, label="Batch")

   # 2. Batch Gradient Descent with regular momentum
    cost_list_momentum = []
    # reinitialize all weights
    W1 = np.random.randn(D, M) / 28
    b1 = np.zeros(M)
    W2 = np.random.randn(M, K) / np.sqrt(M)
    b2 = np.zeros(K)

    cost_list_momentum = []
    mu = 0.9
    dW2 = 0
    db2 = 0
    dW1 = 0
    db1 = 0

    for i in range(int(iter)):
        tempX, tempY = shuffle(Xtrain, Ttrain)
        for j in range(n_batches):
            batchX = tempX[j * batch_size: (j * batch_size + batch_size), :]
            batchT = tempY[j * batch_size: (j * batch_size + batch_size), :]

            Y, Z = myneural.forward(batchX, W1, b1, W2, b2)

            dW2 = mu * dW2 - lr * (myneural.derivative_w2(Z, batchT, Y) + reg * W2)
            W2 += dW2
            db2 = mu * db2 - lr * (myneural.derivative_b2(batchT, Y) + reg * b2)
            b2 += db2
            dW1 = mu * dW1 - lr * (myneural.derivative_w1(batchX, Z, batchT, Y, W2) + reg * W1)
            W1 += dW1
            db1 = mu * db1 - lr * (myneural.derivative_b1(Z, batchT, Y, W2) + reg * b1)
            b1 += db1

            Y, __ = myneural.forward(Xtest, W1, b1, W2, b2)
            singlecost = myneural.cost(Y, Ttest)
            cost_list_momentum.append(singlecost)
            print('Iter : {0} Batch : {1} Cost : {2}'.format(i, j, myneural.cost(Y, Ttest)))
            # if (j % (10) == 0):
            #     print('Accuracy rate : {0}'.format(myneural.error_rate(Y, Ytest)))

    x2 = np.linspace(0, 1, len(cost_list_momentum))
    plt.plot(x2, cost_list_momentum, label="Regular Momentum")

    plt.show()
