import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize

import nn as neuralNet
import utils
import multiClass as mC

def public_Test(X, Y, lambda_):
    # weights = loadmat('data/ex3weights.mat')
    # theta1, theta2 = weights['Theta1'], weights['Theta2']
    # cost = neuralNet.cost(theta1, theta2,X, Y, lambda_)
    # print("Cost: " + cost)

    utils.checkNNGradients(neuralNet.backprop, lambda_)

def show_samples():
    X , Y = readData("ex3data1.mat")
    
    # rand_indices = np.random.choice(X.shape[0], 100, replace=False)
    # utils.displayData(X[rand_indices, :])
    plt.show()

def readData(file):
    data = loadmat('data/ex3data1.mat', squeeze_me=True)

    X_train = data['X']
    y_train = data['y']

    return X_train, y_train

def compareEquals(Y, p):
    acertados = 0
    m = p.shape[0]
    for i in range(m):
        if(Y[i] == p[i]):
            acertados += 1
    
    return (acertados / m) * 100

def encodeY(Y, labels):
    m = len(Y)
    newY = np.zeros((m, labels))
    for i in range(m):
        value = Y[i] 
        newY[i, value] = 1

    return newY

def initiTheta(X, Y, eInit):
    layers_size = 25
    theta1 = np.random.uniform(low= -eInit, high= eInit, size=(layers_size,X.shape[1] + 1))
    theta2 = np.random.uniform(low= -eInit, high= eInit, size=(Y.shape[1],layers_size + 1))

    return theta1, theta2

def learningParameters(X, Y, Y_encoded, lambda_, alpha, iterations):
    theta1, theta2 = initiTheta(X, Y_encoded, 0.12)
    LearnedTheta1, LearnedTheta2_, cost = neuralNet.gradient_descent(X, Y_encoded, theta1, theta2, neuralNet.backprop, alpha, iterations, lambda_)

    result = mC.predict(LearnedTheta1, LearnedTheta2_, X)[3]

    percentage = compareEquals(Y, result)
    
    print(f"Precision: {percentage}%")

def learnParametersSciPy(X, Y, Y_encoded):
    theta1, theta2 = initiTheta(X, Y_encoded, 0.12)

    result = minimize(fun= neuralNet.backprop, x0= theta1, method='TNC', jac=True, options={'maxiter': 100})
    # print(cost(result.x))

def our_test_A():
    X , Y = readData("ex3data1.mat")

    Y_encoded = encodeY(Y, 10)

    lambda_ = 1
    alpha = 1
    iterations = 100
    # public_Test(X, Y, lambda_)
    learningParameters(X, Y, Y_encoded, lambda_, alpha, iterations)



    # print(X.shape)    
    # neuralNet.backprop(theta1, theta2,X, Y, lambda_)
    # When n=2

    # n_label = 10
    # alpha = 0.001
    # print("OneVsAll...")
    # all_theta = mC.oneVsAll(X, Y, n_label, alpha)
    # # print(all_theta)
    # print("Predicting...")
    # p = mC.predictOneVsAll(all_theta, X)
    
    # percentage = compareEquals(Y, p)
    
    # print(f"A: {percentage}%")



def main():
    # show_samples()
    our_test_A()

if __name__ == '__main__':
    main()