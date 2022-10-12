import public_tests as test
import numpy as np
import matplotlib.pyplot as plt
import logistic_reg as lr
import utils as utils

def our_test():
    #read data
    X , Y = readData()
    #initial values
    b_init = -8
    w_init = np.array([0.0, 0.0])
    iterations = 7000
    alpha = 0.001
    #TRAINING
    w , b, history = lr.gradient_descent(X, Y, w_init, b_init,lr.compute_cost, lr.compute_gradient, alpha , iterations)
    #Predict Values
    Y_predicted = lr.predict(X, w, b)
    print(Y_predicted)
    
    #Show values and function
    # utils.plot_data(X, Y, "y_=1", "y_=0", 'green', 'blue')
    utils.plot_decision_boundary(w, b, X, Y)
    
    plt.legend()
    plt.show()

def public_Test():
    # test.sigmoid_test(lr.sigmoid)
    # test.compute_cost_test(lr.compute_cost)
    test.compute_cost_reg_test(lr.compute_cost_reg)

def readData():
    data = np.loadtxt("./data/ex2data1.txt", delimiter=',', skiprows=1)
    X_train = data[:, :2]
    y_train = data[:, 2]

    return X_train, y_train

def main():
    # our_test()
    public_Test()
    # utils.plot_data(X, Y)

if __name__ == '__main__':
    main()