import numpy as np
import logisticReg as lr



#########################################################################
# one-vs-all
#
def oneVsAll(X, y, n_labels, lambda_):
    """
     Trains n_labels logistic regression classifiers and returns
     each of these classifiers in a matrix all_theta, where the i-th
     row of all_theta corresponds to the classifier for label i.

     Parameters
     ----------
     X : array_like
         The input dataset of shape (m x n). m is the number of
         data points, and n is the number of features. 

     y : array_like
         The data labels. A vector of shape (m, ).

     n_labels : int
         Number of possible labels.

     lambda_ : float
         The logistic regularization parameter.

     Returns
     -------
     all_theta : array_like
         The trained parameters for logistic regression for each class.
         This is a matrix of shape (K x n+1) where K is number of classes
         (ie. `n_labels`) and n is number of features without the bias.
     """
    all_theta = np.zeros((n_labels, X.shape[1] + 1))
    
    alpha = 1
    
    for i in range(n_labels):
        newRow =  np.where(y == i, 1, 0)

        w_init = np.zeros(X.shape[1])
        b = 0
        theta_i, b_b, hisotry = lr.gradient_descent(X, newRow, w_init, b, lr.compute_cost_reg, lr.compute_gradient_reg, alpha, 1500, lambda_)
        
        all_theta[i, 0] = b_b
        all_theta[i, 1:] = theta_i

    return all_theta


def predictOneVsAll(all_theta, X):
    """
    Return a vector of predictions for each example in the matrix X. 
    Note that X contains the examples in rows. all_theta is a matrix where
    the i-th row is a trained logistic regression theta vector for the 
    i-th class. You should set p to a vector of values from 0..K-1 
    (e.g., p = [0, 2, 0, 1] predicts classes 0, 2, 0, 1 for 4 examples) .

    Parameters
    ----------
    all_theta : array_like
        The trained parameters for logistic regression for each class.
        This is a matrix of shape (K x n+1) where K is number of classes
        and n is number of features without the bias.

    X : array_like
        Data points to predict their labels. This is a matrix of shape 
        (m x n) where m is number of data points to predict, and n is number 
        of features without the bias term. Note we add the bias term for X in 
        this function. 

    Returns
    -------
    p : array_like
        The predictions for each data point in X. This is a vector of shape (m, ).
    """
    

    # Para cada uno tienes que pasar el 0 de su linea = b
    # Con W, que es el resto de la linea
    # Y con X
    m = len(X)

    p = np.zeros(m)

    #Iterative
    # for i in range(m):
    #     # p[i] = lr.fun_wb(X[i], all_theta[i, 1:])
    #     label = 0
    #     max = 0
    #     for j in range(all_theta.shape[0]):
    #        n_Result = lr.fun_wb(X[i], all_theta[j, 1:], all_theta[j, 0])
    #        if(n_Result > max) : 
    #             max = n_Result
    #             label = j
    #     print(label)
    #     p[i] = label

    # return p
    #Vectorized
    p_ = np.argmax(lr.fun_wb(X , all_theta[:, 1:].T , all_theta[:, 0]), 1)
    # print(np.array_equal(p , p_))
    return p_



#########################################################################
# NN
#
def predict(theta1, theta2, X):
    m = X.shape[0]
    #Input matrix
    X1s = np.c_[np.ones(m), X]
    #First layer
    first = np.dot(theta1, X1s.T)
    #Sigmoid of first multiplication
    X2s = lr.sigmoid(first)
    #new input
    X2s = np.c_[np.ones(len(X2s[0])), X2s.T]
    #second layer
    second = np.dot(theta2, X2s.T)
    #output
    p = lr.sigmoid(second)

    #Final output
    return p.T
