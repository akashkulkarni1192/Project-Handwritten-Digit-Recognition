import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

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


def softmax(a):
    expa = np.exp(a)
    return expa / expa.sum(axis=1, keepdims=True)


def predict(p_y):
    return np.argmax(p_y, axis=1)


def error_rate(p_y, t):
    prediction = predict(p_y)
    return np.mean(prediction != t)
    # same = 0
    # for i in range(len(prediction)):
    #     if(prediction[i] == t[i]):
    #         same = same + 1
    # accuracy = same / len(prediction)
    # return accuracy

def getdata():
    dataframe = pd.read_csv('train.csv')
    data = dataframe.as_matrix().astype(np.float32)
    np.random.shuffle(data)
    X = data[:, 1:]
    Z = normalize_pca(X)
    Y = data[:, 0].astype(np.int32)
    return Z, Y


def forward(X, W, B):
    a = X.dot(W) + B
    return softmax(a)


def gradW(X, Y, T):
    return X.T.dot(T - Y)


def gradB(Y, T):
    return (T - Y).sum(axis=0)


def cost(Y, T):
    return -(T * np.log(Y)).sum()


def batch_gradient_descent(X, T, W, B, N, K, lr, reg, batch_size, Xtest, Ytest, Ttest, iter=50):
    n_batches = int(N / batch_size)
    cost_list = []
    for i in range(iter):
        tempX, tempY = shuffle(X, T)
        for j in range(n_batches):
            batchX = tempX[j * batch_size: (j * batch_size + batch_size), :]
            batchT = tempY[j * batch_size: (j * batch_size + batch_size), :]

            Y = forward(batchX, W, B)
            W += lr * (gradW(batchX, Y, batchT) - reg * W)
            B += lr * (gradB(Y, batchT) - reg * B)
            Y = forward(Xtest, W, B)
            singlecost = cost(Y, Ttest)
            cost_list.append(singlecost)
            print('Iter : {0} Batch : {1} Cost : {2}'.format(i, j, cost(Y, Ttest)))
            if (j % (10) == 0):
                print('Accuracy rate : {0}'.format(error_rate(Y, Ytest)))

    x1 = np.linspace(0, 1, len(cost_list))
    plt.plot(x1, cost_list, label="Batch")
    plt.show()

def normalize(X):
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mu) / std
    return X


if __name__ == '__main__':
    X, Y = getdata()
    X = X[:, :300]

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

    # initializing initial weights
    W = np.random.randn(D, K) / 28
    B = np.zeros(K)

    # initializing hyper parameters
    lr = 0.0001
    reg = 0.01
    batch_size = 500

    batch_gradient_descent(Xtrain, Ttrain, W, B, N, K, lr, reg, batch_size, Xtest, Ytest, Ttest)
    print('End')
