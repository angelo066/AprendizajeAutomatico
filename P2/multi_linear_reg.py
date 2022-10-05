from array import array
from turtle import shape
import numpy as np
import copy
import math


def zscore_normalize_features(X):
    """
    computes  X, zcore normalized by column

    Args:
      X (ndarray (m,n))     : input data, m examples, n features

    Returns:
      X_norm (ndarray (m,n)): input normalized by column
      mu (ndarray (n,))     : mean of each feature
      sigma (ndarray (n,))  : standard deviation of each feature
    """
    X_norm = 0

    if(len(X.shape) > 1):
      X_norm = np.empty((X.shape[0], X.shape[1]))
    else:
      X_norm = np.empty((X.shape[0]))

    mu = np.mean(X , axis = 0)
    sigma = np.std(X , axis = 0)

    for i in range (len(X)):
      X_norm[i] = (X_norm[i] - mu) / sigma
    
    return (X_norm, mu, sigma)

def compute_cost(X, y, w, b):
    """
    compute cost
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
    Returns
      cost (scalar)    : cost
    """
    # cost = 0
    m = y.shape[0]
    #OUR METHOD
    # for i in range(m):
    #   fun = np.dot(w , X[i]) + b
    #   cost += np.sum(fun - y[i]) ** 2 
    # return cost / (2*m)

    return  np.sum(((X @ w + b) - y ) ** 2) / (2 * m )

def compute_gradient(X, y, w, b):
    """
    Computes the gradient for linear regression 
    Args:
      X : (ndarray Shape (m,n)) matrix of examples 
      y : (ndarray Shape (m,))  target value of each example
      w : (ndarray Shape (n,))  parameters of the model      
      b : (scalar)              parameter of the model      
    Returns
      dj_dw : (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w. 
      dj_db : (scalar)             The gradient of the cost w.r.t. the parameter b. 
    """
    # dj_db = 0
    # dj_dw = 0

    m = y.shape[0]

    #OUR METHOD
    # for i in range(m):
    #   fun = np.dot(w , X[i]) + b
    #   dj_dw_i = (fun - y[i]) * X[i]
    #   dj_db_i = (fun - y[i])

    #   dj_db += dj_db_i
    #   dj_dw += dj_dw_i

    # dj_dw = dj_dw / m
    # dj_db = dj_db / m

    # return dj_db, dj_dw

    fun = X @ w + b
    e = fun - y
    dj_db = np.sum(e) / m
    dj_dw = (X.T @ e) / m

    return dj_db , dj_dw  


def gradient_descent(X, y, w_in, b_in, cost_function,
                     gradient_function, alpha, num_iters):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha

    Args:
      X : (array_like Shape (m,n)    matrix of examples 
      y : (array_like Shape (m,))    target value of each example
      w_in : (array_like Shape (n,)) Initial values of parameters of the model
      b_in : (scalar)                Initial value of parameter of the model
      cost_function: function to compute cost
      gradient_function: function to compute the gradient
      alpha : (float) Learning rate
      num_iters : (int) number of iterations to run gradient descent
    Returns
      w : (array_like Shape (n,)) Updated values of parameters of the model
          after running gradient descent
      b : (scalar)                Updated value of parameter of the model 
          after running gradient descent
      J_history : (ndarray): Shape (num_iters,) J at each iteration,
          primarily for graphing later
    """
    m = y.shape[0]
    
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in
    
    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w, b)
        
        w -= alpha * dj_dw
        b -= alpha * dj_db

        if i < 100000:
            cost = cost_function(X,y,w,b)
            J_history.append(cost)
    
    return w, b, J_history
