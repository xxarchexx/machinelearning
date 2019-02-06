import numpy as np
import math

#возвращает скаляр - вероятность
def Sigmoid(z):
    return float(1)/float(1 + (math.exp(z * -1.0)))

#h(x)
def Hipothesis(X,theta):
    z = 0
    for i in range(len(theta)):
        z += theta[i] * X[i]
    Sigmoid(z

#функция ошибки оценивает ошибку модели при текущих параметрах theta
def Cost_Function(X,Y, theta,m):
    sumOfErrors: int = 0
    for i in range(len(m)):
        hi = Hipothesis(X,theta)
        if Y[i] == 0:
            error = (1-Y[i]) * (math.log(1 - hi))
        elif Y[i] == 1:
            error = Y[i]*math.log(hi)
        sumOfErrors += error
    J = -(1/m) * sumOfErrors
    return J


#град спуск - идем по параметрам theta вычисляем ошибку на определенном цикле theta
#для этого  идем по всей модели считаем ошибку и берем производную.
#заменяем текущий theta на новый
def Gdadient_descent(X, Y, m, alpha, theta):
    #old_theta = theta[i]
    new_thetas = []
    for i in range(len(theta)):
        old_theta = theta[i]
        sumOfError = 0
        for ii in range(len(m)):
            xi = X[ii]
            error = (Hipothesis(xi[i], theta) - Y[ii]) * xi[i]
            sumOfError += error
        new_theta = old_theta - alpha * 1/m * sumOfError
        new_thetas.append(new_theta)
    return new_theta

initial_theta = [0,0]
alpha = 0.1
iterations = 1000

#главная функция передаем  - передаем начальные параметры , дальше она стартует град спуск
def LogRegression(X, Y, iterations,  alpha, theta):
    for i in range(len(iterations)):
        new_theta = Gdadient_descent(X, Y, iterations, alpha, theta)
        theta = new_theta
        if i % 100 == 0:
            print(Cost_Function(X, Y, theta, len(y)))