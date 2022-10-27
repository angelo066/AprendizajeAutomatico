import numpy as np
import matplotlib.pyplot as plt
# import utils as utils
from scipy.io import loadmat

import nn as neuralNet

import logisticReg as lr
# import public_tests as test

# def public_Test():
#     test.sigmoid_test(lr.sigmoid)
#     test.compute_cost_test(lr.compute_cost)
#     test.compute_gradient_test(lr.compute_gradient)
#     test.compute_cost_reg_test(lr.compute_cost_reg)
#     test.compute_gradient_reg_test(lr.compute_gradient_reg)

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

def our_test_A():
    X , Y = readData("ex3data1.mat")

    Y = encodeY(Y, 10)

    weights = loadmat('data/ex3weights.mat')
    theta1, theta2 = weights['Theta1'], weights['Theta2']

    lambda_ = 0.001
    cost = neuralNet.cost(theta1, theta2,X, Y, lambda_)
    print(cost)
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