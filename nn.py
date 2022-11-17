import numpy as np
import scipy.io as spio
import scipy.optimize as spopt
import utils
from matplotlib import pyplot
import logistic_reg as lreg

def feedforward(theta1, theta2, X):
    m = len(X)

    X1s = np.c_[np.ones(m), X]
    
    z2 = np.dot(theta1, X1s.T)

    a2 = lreg.sigmoid(z2)

    a2s = np.vstack([np.ones((1, a2.shape[1])), a2])

    z3 = np.dot(theta2, a2s)

    hThetaX = lreg.sigmoid(z3)

    return hThetaX, X1s, a2s

def cost(theta1, theta2, X, y, lambda_):
    """
    Compute cost for 2-layer neural network. 

    Parameters
    ----------
    theta1 : array_like
        Weights for the first layer in the neural network.
        It has shape (2nd hidden layer size x input size + 1)

    theta2: array_like
        Weights for the second layer in the neural network. 
        It has shape (output layer size x 2nd hidden layer size + 1)

    X : array_like
        The inputs having shape (number of examples x number of dimensions).

    y : array_like
        1-hot encoding of labels for the input, having shape 
        (number of examples x number of labels).

    lambda_ : float
        The regularization parameter. 

    Returns
    -------
    J : float
        The computed value for the cost function. 

    """
    m = X.shape[0]

    predict = feedforward(theta1, theta2, X)[0]

    predict = predict.T

    J = (-1/m)* np.sum((y * np.log(predict)) + (1-y) * np.log(1-predict))

    J += (lambda_ / (2*m)) * (np.sum(theta1[:, 1:]**2) + np.sum(theta2[:, 1:]**2))

    return J

# def backprop_aux(p):
#     theta1 = np.reshape(p[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1)))
#     theta2 = np.reshape(p[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1)))
#     J, grad1, grad2 = backprop(theta1, theta2, X, y, reg)
#     grad = np.concatenate((np.ravel(grad1), np.ravel(grad2)))
# return J, grad

def backprop(theta1, theta2, X, y, lambda_):
    """
    Compute cost and gradient for 2-layer neural network. 

    Parameters
    ----------
    theta1 : array_like
        Weights for the first layer in the neural network.
        It has shape (2nd hidden layer size x input size + 1)

    theta2: array_like
        Weights for the second layer in the neural network. 
        It has shape (output layer size x 2nd hidden layer size + 1)

    X : array_like
        The inputs having shape (number of examples x number of dimensions).

    y : array_like
        1-hot encoding of labels for the input, having shape 
        (number of examples x number of labels).

    lambda_ : float
        The regularization parameter. 

    Returns
    -------
    J : float
        The computed value for the cost function. 

    grad1 : array_like
        Gradient of the cost function with respect to weights
        for the first layer in the neural network, theta1.
        It has shape (2nd hidden layer size x input size + 1)

    grad2 : array_like
        Gradient of the cost function with respect to weights
        for the second layer in the neural network, theta2.
        It has shape (output layer size x 2nd hidden layer size + 1)

    """

    J = cost(theta1, theta2, X, y, lambda_)

    grad1 = np.zeros(theta1.shape)
    grad2 = np.zeros(theta2.shape)

    m = X.shape[0]

    for i in range(m):
        a3, a1, a2 = feedforward(theta1, theta2, [X[i]])

        d3 = a3.T-y[i]

        gprimez2 = a2* (1 - a2)

        d2 = np.dot(theta2.T, d3.T) * gprimez2

        d2 = d2[1:]

        grad1 += np.dot(d2,a1)
        grad2 += np.dot(d3.T,a2.T)

    grad1[:,1:] = (grad1[:,1:]+lambda_*theta1[:,1:])/m
    grad2[:,1:] = (grad2[:,1:]+lambda_*theta2[:,1:])/m


    grad1[:,0] = grad1[:,0]/m
    grad2[:,0] = grad2[:,0]/m

    return J , grad1, grad2

def backpropMinimize(thetas, X, y, lambda_):
    layers = 25
    m = y.shape[1]
    n = X.shape[1]

    theta1 = np.reshape(thetas[:layers * (n+1)], (layers, n+1))
    theta2 = np.reshape(thetas[layers * (n+1):], (m, layers+1))

    J = cost(theta1, theta2, X, y, lambda_)

    grad1 = np.zeros(theta1.shape)
    grad2 = np.zeros(theta2.shape)

    m = X.shape[0]

    for i in range(m):
        a3, a1, a2 = feedforward(theta1, theta2, [X[i]])

        d3 = a3.T-y[i]

        gprimez2 = a2* (1 - a2)

        d2 = np.dot(theta2.T, d3.T) * gprimez2

        d2 = d2[1:]

        grad1 += np.dot(d2,a1)
        grad2 += np.dot(d3.T,a2.T)

    grad1[:,1:] = (grad1[:,1:]+lambda_*theta1[:,1:])/m
    grad2[:,1:] = (grad2[:,1:]+lambda_*theta2[:,1:])/m


    grad1[:,0] = grad1[:,0]/m
    grad2[:,0] = grad2[:,0]/m

    return J , np.concatenate([np.ravel(grad1), np.ravel(grad2)])

def backpropCheck():
    data = spio.loadmat('data/ex3data1.mat', squeeze_me=True)

    n_labels = 10

    X = data['X']
    y = data['y']

    encodedY = np.zeros((y.shape[0], n_labels))

    for i in range(y.shape[0]):
        encodedY[i, y[i]] = 1

    weights = spio.loadmat('data/ex3weights.mat')
    theta1, theta2 = weights['Theta1'], weights['Theta2']

    utils.checkNNGradients(backprop,0.5)

def learningParams(num_iters, alpha, lambda_):
    data = spio.loadmat('data/ex3data1.mat', squeeze_me=True)

    n_labels = 10

    X = data['X']
    y = data['y']

    encodedY = np.zeros((y.shape[0], n_labels))

    for i in range(y.shape[0]):
        encodedY[i, y[i]] = 1

    INIT_EPSILON = 0.12

    layers = 25
    m = encodedY.shape[1]
    n = X.shape[1]

    theta1 = np.random.random((layers, n+1))*(2*INIT_EPSILON) - INIT_EPSILON
    theta2 = np.random.random((m, layers+1))*(2*INIT_EPSILON) - INIT_EPSILON

    # for i in range(num_iters):
    #     J , grad1, grad2 = backprop(theta1, theta2, X, encodedY, lambda_)
    #     theta1 -= alpha*grad1
    #     theta2 -= alpha*grad2

    thetas = np.concatenate([theta1.ravel(), theta2.ravel()])

    result = spopt.minimize(fun=backpropMinimize, x0=thetas, args=(X, encodedY, lambda_), method='TNC', jac=True, options={'maxiter': num_iters})
    
    theta1 = np.reshape(result.x[:layers * (n+1)], (layers, n+1))
    theta2 = np.reshape(result.x[layers * (n+1):], (m, layers+1))

    a3 = feedforward(theta1, theta2, X)[0]
    p = np.argmax(a3, 0)

    equal = 0

    for i in range(y.shape[0]):
        if p[i] == y[i]:
            equal += 1


    percentage = (str)(100*(equal/y.shape[0]))
    print("Porcentaje de acierto: " + percentage + "%")
        

def main():
    #backpropCheck()
    learningParams(100, 1, 1)

main()