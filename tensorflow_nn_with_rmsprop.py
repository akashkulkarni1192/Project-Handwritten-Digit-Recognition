import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import nn_util as my_util



def main():
    X, Y = my_util.get_data()

    max_iter = 20
    print_period = 10
    lr = 0.0004
    reg = 0

    Xtrain = X[:-1000, ]
    Ytrain = Y[:-1000]
    Xtest = X[-1000:, ]
    Ytest = Y[-1000:]

    Ytrain_enc = my_util.one_hot_encoding(Ytrain)
    Ytest_enc = my_util.one_hot_encoding(Ytest)

    N, D = Xtrain.shape
    batch_size = 500
    n_batches = N / batch_size

    M = 300
    K = 10

    W1_init = np.random.randn(D, M) / 28
    b1_init = np.zeros(M)
    W2_init = np.random.randn(M, K) / np.sqrt(M)
    b2_init = np.zeros(K)

    thX = tf.placeholder(tf.float32, shape=(None, D), name='X')
    thT = tf.placeholder(tf.float32, shape=(None, K), name='T')

    W1 = tf.Variable(initial_value=W1_init.astype(np.float32))
    b1 = tf.Variable(initial_value=b1_init.astype(np.float32))
    W2 = tf.Variable(initial_value=W2_init.astype(np.float32))
    b2 = tf.Variable(initial_value=b2_init.astype(np.float32))

    thZ = tf.nn.relu(tf.matmul(thX, W1) + b1)
    thY = tf.nn.softmax(tf.matmul(thZ, W2) + b2)

    prediction_operation = tf.argmax(thY, axis=1)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=thY, labels=thT))

    train_operation = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        costs = []
        for i in range(max_iter):
            for j in range(int(n_batches)):
                Xbatch = Xtrain[j * batch_size:(j * batch_size + batch_size), :]
                Ybatch = Ytrain_enc[j * batch_size:(j * batch_size + batch_size), :]

                sess.run(train_operation, feed_dict={thX: Xbatch, thT: Ybatch})

                if (j % print_period == 0):
                    cost_val = sess.run(cost, feed_dict={thX: Xtest, thT: Ytest_enc})
                    prediction_val = sess.run(prediction_operation, feed_dict={thX: Xtest})
                    error_val = my_util.error_rate(prediction_val, Ytest)
                    print('cost : {0} error = {1}'.format(cost_val, error_val))
                    costs.append(cost_val)

    plt.plot(costs)
    plt.show()


if __name__ == '__main__':
    # main()
    write_csv()