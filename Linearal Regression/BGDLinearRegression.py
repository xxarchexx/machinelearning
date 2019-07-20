# calling the principal function with learning_rate = 0.0001 and
# num_iters = 300000

def BGD(theta, alpha, num_iters, h, X, y, n):
    cost = np.ones(num_iters)
    for i in range(0, num_iters):
        theta[0] = theta[0] - (alpha / X.shape[0]) * sum(h - y)
        for j in range(1, n + 1):
            theta[j] = theta[j] - (alpha / X.shape[0]) * sum((h - y) * X.transpose()[j])
    h = hypothesis(theta, X, n)
    cost[i] = (1 / X.shape[0]) * 0.5 * sum(np.square(h - y))


theta = theta.reshape(1, n + 1)
return theta, cost


def hypothesis(theta, X, n):
    h = np.ones((X.shape[0], 1))
    theta = theta.reshape(1, n + 1)
    for i in range(0, X.shape[0]):
        h[i] = float(np.matmul(theta, X[i]))
    h = h.reshape(X.shape[0])
    return h


def linear_regression(X, y, alpha, num_iters):
    n = X.shape[1]
    one_column = np.ones((X.shape[0], 1))
    X = np.concatenate((one_column, X), axis=1)
    # initializing the parameter vector...
    theta = np.zeros(n + 1)
    # hypothesis calculation....
    h = hypothesis(theta, X, n)
    # returning the optimized parameters by Gradient Descent...
    theta, cost = BGD(theta, alpha, num_iters, h, X, y, n)
    return theta, cost

theta, cost = linear_regression(X_train, y_train,
                                               0.0001, 300000)

data = np.loadtxt('data2.txt', delimiter=',')
X_train = data[:,[0,1]] #feature set
y_train = data[:,2] #label set

mean = np.ones(X_train.shape[1])
std = np.ones(X_train.shape[1])

import matplotlib.pyplot as plt
cost = list(cost)
n_iterations = [x for x in range(1,300001)]
plt.plot(n_iterations, cost)
plt.xlabel('No. of iterations')
plt.ylabel('Cost')

for i in range(0, X_train.shape[1]):
    mean[i] = np.mean(X_train.transpose()[i])
    std[i] = np.std(X_train.transpose()[i])
    for j in range(0, X_train.shape[0]):
        X_train[j][i] = (X_train[j][i] - mean[i])/std[i]
