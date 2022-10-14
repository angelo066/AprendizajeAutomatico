from asyncio.windows_events import NULL
from cmath import log
from re import M
import numpy as np
import copy
import math

def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z

    """
    return  1/(1 + np.exp(-z))

#########################################################################
# logistic regression
#
def fun_wb(X, w, b):
  return sigmoid(X @ w + b)

def compute_cost(X, y, w, b, lambda_=None):
    """
    Computes the cost over all examples
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (array_like Shape (m,)) target value
      w : (array_like Shape (n,)) Values of parameters of the model
      b : scalar Values of bias parameter of the model
      lambda_: unused placeholder
    Returns:
      total_cost: (scalar)         cost
    """
    total_cost = 0
    m = y.shape[0]

    # fun_val = fun_wb(X,w, b)
    # total_cost = np.sum((y + np.log(fun_val)) - (1 - y) * np.log(1 - fun_val))
    
    for i in range(m):
      fun_val = fun_wb(X[i],w, b)

      log_cost = np.log(fun_val)
      log_cost2 = np.log(1 - fun_val)

      total_cost += (-y[i] * log_cost) -  (1 - y [i]) * log_cost2

    return (total_cost/m)

def compute_gradient(X, y, w, b, lambda_=None):
    dj_dw = 0  
    dj_db = 0

    m = y.shape[0]

    #en teoria hay que hacer un doble for
    for i in range(m):
      dj_dw += (fun_wb(X[i],w,b) - y[i]) * X[i]
      dj_db += (fun_wb(X[i],w,b) - y[i])

    return dj_db / m, dj_dw / m


#########################################################################
# regularized logistic regression
#
def compute_cost_reg(X, y, w, b, lambda_=1):
    """
    Computes the cost over all examples
    Args:
      X : (array_like Shape (m,n)) data, m examples by n features
      y : (array_like Shape (m,)) target value 
      w : (array_like Shape (n,)) Values of parameters of the model      
      b : (array_like Shape (n,)) Values of bias parameter of the model
      lambda_ : (scalar, float)    Controls amount of regularization
    Returns:
      total_cost: (scalar)         cost 
    """
    total_cost = 0
    m = y.shape[0]
    
    for i in range(m):
      
      fun_val = fun_wb(X[i],w, b)

      log_cost = np.log(fun_val)
      log_cost2 = np.log(1 - fun_val)

      total_cost += (-y[i] * log_cost) -  (1 - y [i]) * log_cost2
      
    total_cost = total_cost / m
    
    w_cost = np.sum(w**2)
      
    w_cost = (w_cost * lambda_) / (2 * m)
     
    total_cost += w_cost 
    
    return total_cost


def compute_gradient_reg(X, y, w, b, lambda_=1):
    """
    Computes the gradient for linear regression 

    Args:
      X : (ndarray Shape (m,n))   variable such as house size 
      y : (ndarray Shape (m,))    actual value 
      w : (ndarray Shape (n,))    values of parameters of the model      
      b : (scalar)                value of parameter of the model  
      lambda_ : (scalar,float)    regularization constant
    Returns
      dj_db: (scalar)             The gradient of the cost w.r.t. the parameter b. 
      dj_dw: (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w. 

    """
    dj_db = 0 
    dj_dw = 0
    m = y.shape[0]
    
    for i in range(m):
      dj_db += fun_wb(X[i], w , b) - y[i]
      dj_dw += (fun_wb(X[i], w, b) - y[i]) * X[i]
    
    return dj_db / m, (dj_dw / m) + (np.dot(np.divide(lambda_,m),w))


#########################################################################
# gradient descent
#
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_=None):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha

    Args:
      X :    (array_like Shape (m, n)
      y :    (array_like Shape (m,))
      w_in : (array_like Shape (n,))  Initial values of parameters of the model
      b_in : (scalar)                 Initial value of parameter of the model
      cost_function:                  function to compute cost
      alpha : (float)                 Learning rate
      num_iters : (int)               number of iterations to run gradient descent
      lambda_ (scalar, float)         regularization constant

    Returns:
      w : (array_like Shape (n,)) Updated values of parameters of the model after
          running gradient descent
      b : (scalar)                Updated value of parameter of the model after
          running gradient descent
      J_history : (ndarray): Shape (num_iters,) J at each iteration,
          primarily for graphing later
    """
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


#########################################################################
# predict
#
def predict(X, w, b):
    """
    Predict whether the label is 0 or 1 using learned logistic
    regression parameters w and b

    Args:
    X : (ndarray Shape (m, n))
    w : (array_like Shape (n,))      Parameters of the model
    b : (scalar, float)              Parameter of the model

    Returns:
    p: (ndarray (m,1))
        The predictions for X using a threshold at 0.5
    """
    if(fun_wb(X, w, b)> 0.5):
      return 1

    return 0