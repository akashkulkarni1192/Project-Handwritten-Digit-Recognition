import scipy.io as sciIO
import numpy as numpu
import scipy.optimize as optu


def sigmoid(x):
    res = 1 / (1 + numpu.exp(-1 * x))
    return res


def cost(theta, x, y, lambu):
    m, n = x.shape
    t1 = numpu.log(sigmoid(x.dot(theta)))
    t2 = numpu.log(sigmoid(x.dot(theta)))

    t = numpu.multiply(-y, t1) - numpu.multiply(1 - y, t2)
    reg = lambu / (2 * m) + numpu.sum(numpu.power(theta, 2))
    cost_j = -(numpu.sum(t) / m) + reg
    return cost_j


def step_gradient(theta, x_matrix, y_matrix, lambu):
    m = len(x_matrix)
    n = len(x_matrix[0])
    temp_theta = numpu.copy(theta)
    temp_theta = numpu.reshape(temp_theta, (n, 1))
    hypo_matrix = sigmoid(numpu.dot(x_matrix, temp_theta))
    y_matrix = numpu.reshape(y_matrix, (m, 1))
    diff = numpu.subtract(hypo_matrix, y_matrix)
    # grad = numpu.zeros(shape=(len(temp_theta), 1))
    # grad[0][0] = 1 / m * numpu.sum(
    #     numpu.multiply(numpu.subtract(hypo_matrix, y_matrix), numpu.row_stack(x_matrix[:, 0])))

    grad = 1 / m * numpu.dot(diff.transpose(), x_matrix).transpose() + lambu / m * temp_theta
    grad[0][0] = 1 / m * numpu.sum(
        numpu.multiply(numpu.subtract(hypo_matrix, y_matrix), numpu.row_stack(x_matrix[:, 0])))

    # for j in range(1, len(temp_theta)):
    #     #print(j)
    #     grad[j][0] = 1 / m * numpu.sum(
    #         numpu.multiply(numpu.subtract(hypo_matrix, y_matrix), numpu.row_stack(x_matrix[:, j]))) + lambu / m * \
    #                                                                                                   temp_theta[j][0]

    return grad


def predict(x_matrix, theta):
    (m, n) = x_matrix.shape

    # theta = theta.transpose()
    hypo_matrix = sigmoid(numpu.dot(x_matrix, theta.transpose()))

    res = numpu.argmax(hypo_matrix, axis=1)
    res = res + numpu.ones(shape=res.shape)

    print('s')
    return res


def test_case():
    x_temp = [x for x in range(1, 16)]
    X_t = numpu.column_stack([numpu.ones(shape=(5, 1)), (numpu.reshape(x_temp, (3, 5))).transpose() / 10]);
    y_t = numpu.reshape([1, 0, 1, 0, 1], (5,1))
    theta_t = numpu.reshape([-2, -1, 1, 2], (4,1))
    lambda_t = 3;
    grad = step_gradient(theta_t, X_t, y_t, lambda_t)
    print('bubu playing guitar')


def run():
    test_case()
    mat_data = sciIO.loadmat('ex3data1.mat')

    X = mat_data['X']
    (m, n) = X.shape
    k = 10
    x_matrix = numpu.column_stack([numpu.ones(shape=(m, 1)), X])
    Y = mat_data['y']
    initial_theta = numpu.zeros(shape=(k, n + 1))

    y_matrix = [1 if x == Y[0, 0] else 0 for x in range(1, 11)]

    for i in range(1, m):
        new_row = [1 if x == Y[i, 0] else 0 for x in range(1, 11)]
        y_matrix = numpu.vstack((y_matrix, new_row))
    lambda_val = 0.003

    # truncated Newton algorithm
    learning_rate = 0.01

    final_theta = numpu.zeros(shape=(k, n + 1))
    for i in range(k):
        single_theta = numpu.zeros(shape=(1, n + 1))
        fmin = optu.minimize(fun=cost, x0=single_theta, args=(x_matrix, y_matrix[:, i], learning_rate), method='TNC',
                              jac=step_gradient)

        final_theta[i] = fmin.x
        #fmin = optu.fmin_cg(f=cost, x0=single_theta, fprime=step_gradient,
                            # args=(x_matrix, y_matrix[:, i], learning_rate))
    print(final_theta)
    y_pred = predict(x_matrix, final_theta)
    y_pred = numpu.reshape(y_pred, Y.shape)
    correct = [1 if a == b else 0 for (a, b) in zip(y_pred, Y)]

    accuracy = (sum(map(int, correct)) / float(len(correct)))
    print(accuracy)


if __name__ == '__main__':
    run()
