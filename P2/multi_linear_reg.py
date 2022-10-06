from array import array
from turtle import shape
import numpy as np
import copy
import math


def zscore_normalize_features(X):
    X_norm = 0

    X_norm = np.empty((X.shape[0], X.shape[1]))
    # if(len(X.shape) > 1):
    # else:
    #   X_norm = np.empty((X.shape[0]))

    mu = np.mean(X , axis = 0)
    sigma = np.std(X , axis = 0)
    
    # X_norm = (X - mu) / sigma
    for i in range (len(X)):
      X_norm[i] = (X[i] - mu) / sigma
    return (X_norm, mu, sigma)

def compute_cost(X, y, w, b):
    m = y.shape[0]
    return  np.sum(((X @ w + b) - y ) ** 2) / (2 * m )

def compute_gradient(X, y, w, b):
    m = y.shape[0]

    fun = X @ w + b
    e = fun - y
    dj_db = np.sum(e / m) 
    dj_dw = (X.T @ e) / m

    return dj_db , dj_dw  

def gradient_descent(X, y, w_in, b_in, cost_function,
                     gradient_function, alpha, num_iters):
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
