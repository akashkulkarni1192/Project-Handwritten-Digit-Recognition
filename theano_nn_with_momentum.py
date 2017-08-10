import numpy as np
import pandas as pd
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import nn_util as my_util


def main():
    X, Y = my_util.get_data('train.csv')

    max_iter = 20
    print_period = 10
    lr = 0.00004
    reg = 0

    Xtrain = X[:-1000, ]
    Ytrain = Y[:-1000]
    Xtest = X[-1000:, ]
    Ytest = Y[-1000:]

    Ytrain_enc = my_util.one_hot_encoding(Ytrain)
    Ytest_enc = my_util.one_hot_encoding(Ytest)

    N, D = Xtrain.shape
    batch_size = 50
    n_batches = N / batch_size

    M = 300
    K = 10

    W1_init = np.random.randn(D, M) / 28
    b1_init = np.zeros(M)
    W2_init = np.random.randn(M, K) / np.sqrt(M)
    b2_init = np.zeros(K)

    thX = T.matrix('X')
    thT = T.matrix('T')

    W1 = theano.shared(W1_init, 'W1')
    b1 = theano.shared(b1_init, 'b1')
    W2 = theano.shared(W2_init, 'W2')
    b2 = theano.shared(b2_init, 'b2')

    thZ = T.nnet.relu(thX.dot(W1) + b1)
    thY = T.nnet.softmax(thZ.dot(W2) + b2)

    cost = -(thT * T.log(thY)).sum() + reg * ((W1 * W1).sum() + (b1 * b1).sum() + (W2 * W2).sum() + (b2 * b2).sum())
    prediction = T.argmax(thY, axis=1)

    update_W1 = W1 - lr * T.grad(cost, W1)
    update_b1 = b1 - lr * T.grad(cost, b1)
    update_W2 = W2 - lr * T.grad(cost, W2)
    update_b2 = b2 - lr * T.grad(cost, b2)

    train = theano.function(
        inputs=[thX, thT],
        updates=[(W1, update_W1), (b1, update_b1), (W2, update_W2), (b2, update_b2)]
    )

    get_prediction = theano.function(
        inputs=[thX, thT],
        outputs=[cost, prediction]
    )

    costs = []
    for i in range(max_iter):
        for j in range(int(n_batches)):
            Xbatch = Xtrain[j * batch_size:(j * batch_size + batch_size), :]
            Ybatch = Ytrain_enc[j * batch_size:(j * batch_size + batch_size), :]

            train(Xbatch, Ybatch)

            if (j % print_period == 0):
                cost_val, prediction_val = get_prediction(Xtest, Ytest_enc)
                error_val = my_util.error_rate(prediction_val, Ytest)
                print('cost : {0} error = {1}'.format(cost_val, error_val))
                costs.append(cost_val)

    plt.plot(costs)
    plt.show()


if __name__ == '__main__':
    main()
