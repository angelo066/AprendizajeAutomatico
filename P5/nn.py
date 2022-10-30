import numpy as np
import multiClass as mC
import logisticReg as lr


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
    J = 0
    m = y.shape[0]

    h0 = mC.predict(theta1, theta2, X)
    fun_val = h0

    log_cost = np.log(fun_val)
    log_cost2 = np.log(1 - fun_val)
    J += np.sum((-y * log_cost) -  (1 - y) * log_cost2)

    J = J /m

    reg_tem = lambda_/(2*m) * ((theta1[1:, 1:]** 2).sum() + (theta2[1:, 1:]** 2).sum() )

    return J + reg_tem

def g_dev(z, a): 
    a * (np.ones(a.shape[0]) - a)


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
    
    m = X.shape[0]
    #Input matrix
    a1 = np.c_[np.ones(m), X]
    #First layer
    z2 = np.dot(theta1, a1.T)
    #Sigmoid of first multiplication
    a2 = lr.sigmoid(z2)
    #new input
    a2 = np.c_[np.ones(len(a2[0])), a2.T]
    #second layer
    z3 = np.dot(theta2, a2.T) #-> a4
    #output
    a3 = lr.sigmoid(z3)
    
    a3 = a3
    
    J = 0

    grad1 = 0
    grad2 = 0

    
    error_3 = a3.T - y

    weight_error = theta2.T @ error_3.T
    error_2 = weight_error * g_dev(z3, a3)

    print(error_2)

    return (J, grad1, grad2)

