import numpy as np
import multiClass as mC
import logisticReg as lr
import copy

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

    #Predecimos
    h0 = mC.feedForward(theta1, theta2, X)[0]
    fun_val = h0

    log_cost = np.log(fun_val)
    log_cost2 = np.log(1 - fun_val)
    #Computamos el coste por cada capa de la red neuronal.
    J += np.sum((-y * log_cost) -  (1 - y) * log_cost2)

    #Dividimos por el numero de datos
    J = J /m

    #Añadimos el termino de regularizacion
    reg_tem = lambda_/(2*m) * (np.sum(theta1[:, 1:]** 2) + np.sum(theta2[:, 1:]** 2) )

    return J + reg_tem

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
    grad1 = grad2 = 0

    m = X.shape[0]

    J = cost(theta1, theta2,X, y, lambda_)

    #Predicción inicial
    a3, a2, a1, p = mC.feedForward(theta1, theta2, X)

    for i in range(m):
        error_3 = a3[i] - y[i]
        g_primeZ = a2[i] * (np.ones(a2[i].shape)- a2[i])

        error_2 = np.dot(theta2.T,error_3) * g_primeZ.T
        error_2 = error_2[1:] #eliminar primera columna
        
        #Entrenamos los pesos
        grad1 += np.matmul(error_2[:, np.newaxis], a1[i][np.newaxis, :])
        grad2 += np.matmul(error_3[:, np.newaxis], a2[i][np.newaxis, :])

    #Dividimos por el número de datos y aplicamos lambda sobre cada peso
    #omitimos la primera columna ya que dicha es el termino independiente B.
    grad1[:, 1:] = (1/m) * grad1[:,1:] + (1/m)*lambda_*theta1[:, 1:]
    grad1[:, 0] = (1/m) * grad1[:, 0]

    grad2[:, 1:] = (1/m) * grad2[:,1:] + (1/m)*lambda_*theta2[:, 1:]
    grad2[:, 0] = (1/m) * grad2[:, 0]

    return (J, grad1, grad2)


def gradient_descent(X, y, theta1_in, theta2_in, gradient_function, alpha, num_iters, lambda_=None):
    J_history = []
    theta1 = copy.deepcopy(theta1_in)
    theta2 = theta2_in
    
    for i in range(num_iters):
      cost, dj_dj1, dj_dj2 = gradient_function(theta1, theta2, X, y,lambda_)
      theta1 -= alpha * dj_dj1
      theta2 -= alpha * dj_dj2

      if i < 100000:
        J_history.append(cost)

    return theta1, theta2, J_history
