import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize

import nn as neuralNet
import utils
import multiClass as mC

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

def initTheta(X, Y, eInit):
    layers_size = 25
    theta1 = np.random.uniform(low= -eInit, high= eInit, size=(layers_size,X.shape[1] + 1))
    theta2 = np.random.uniform(low= -eInit, high= eInit, size=(Y.shape[1],layers_size + 1))

    return theta1, theta2

def learningParameters(X, Y, Y_encoded, lambda_, alpha, iterations):
    theta1, theta2 = initTheta(X, Y_encoded, 0.12)
    LearnedTheta1, LearnedTheta2_, cost = neuralNet.gradient_descent(X, Y_encoded, theta1, theta2, neuralNet.backprop, alpha, iterations, lambda_)

    result = mC.feedForward(LearnedTheta1, LearnedTheta2_, X)[3]

    percentage = compareEquals(Y, result)
    
    print(f"Precision: {percentage}%")

def backpropAuxForMinimize(thetas, layers, X, Y, lambda_):
    m = Y.shape[1]
    n = X.shape[1]

    #Desenrrollamos para poder utilizar nuestra funcion
    theta1 = np.reshape(thetas[:layers * (n+1)], (layers, n+1))
    theta2 = np.reshape(thetas[layers * (n+1):], (m, layers+1))

    J, grad1, grad2 = neuralNet.backprop(theta1, theta2, X, Y, lambda_)

    return J , np.concatenate([np.ravel(grad1), np.ravel(grad2)])

def learnParametersSciPy(X, Y, Y_encoded, lambda_, num_iters):
    theta1, theta2 = initTheta(X, Y_encoded, 0.12)

    layers = 25
    m = Y_encoded.shape[1]
    n = X.shape[1]

    #Enrrollamos los pesos en un array unidimensional
    thetas = np.concatenate([theta1.ravel(), theta2.ravel()])

    #Pasmos la función a minimizar (BackPropagation) y sus datos
    result = minimize(fun=backpropAuxForMinimize, x0=thetas, args=(layers, X, Y_encoded, lambda_), method='TNC', jac=True, options={'maxiter': num_iters})

    #Desenrrollamos para poder realizar una prediccion sobre los pesos resultado.
    theta1 = np.reshape(result.x[:layers * (n+1)], (layers, n+1))
    theta2 = np.reshape(result.x[layers * (n+1):], (m, layers+1))

    result = mC.feedForward(theta1, theta2, X)[3]
    percentage = compareEquals(Y, result)
    print(f"Precision: {percentage}%")

def our_test_A():
    X , Y = readData("ex3data1.mat")

    Y_encoded = encodeY(Y, 10)

    lambda_ = 1
    alpha = 1
    utils.checkNNGradients(neuralNet.backprop)
    learningParameters(X, Y, Y_encoded, lambda_, alpha, 1000)
    # learnParametersSciPy(X, Y, Y_encoded, lambda_, 100) #con 1000 da una precision de 99.64%

def main():
    # show_samples()
    our_test_A()

if __name__ == '__main__':
    main()