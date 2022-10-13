import public_tests as test
import numpy as np
import matplotlib.pyplot as plt
import logistic_reg as lr
import utils as utils

def our_test_A():
    #read data
    X , Y = readData("ex2data1.txt")
    #initial values
    b_init = 1
    w_init = np.array([0.0, 0.0])
    iterations = 10000
    alpha = 0.01
    #TRAINING
    w , b, history = lr.gradient_descent(X, Y, w_init, b_init,lr.compute_cost_reg, lr.compute_gradient_reg, alpha , iterations)
    #Predict Values
    
    correct_ones = 0
    for i in range(Y.shape[0]):
        if(Y[i] == lr.predict(X[i], w,b)):
            correct_ones +=1
    
    print(f"\n The accuracy is {(correct_ones/Y.shape[0])*100} % \n")

    #Show values and function
    utils.plot_data(X, Y, "y_=1", "y_=0", 'green', 'blue')
    utils.plot_decision_boundary(w, b, X, Y)
    
    plt.legend()
    plt.show()

def our_test_B():
    #read data
    X , Y = readData("ex2data2.txt")
    #initial values
    b_init = 1
    iterations = 1000000
    alpha = 0.01

    #TRAINING
    #X[:, 0] -> : todas las filas de la columna 0

    X_stack = utils.map_feature(X[:,0],X[:,1])
    w_init = np.zeros(X_stack.shape[1])
 
    w , b, history = lr.gradient_descent(X_stack, Y, w_init, b_init,lr.compute_cost_reg, lr.compute_gradient_reg, alpha , iterations)

    correct_ones = 0
    for i in range(Y.shape[0]):
        if(Y[i] == lr.predict(X_stack[i], w,b)):
            correct_ones +=1
    
    print(f"\n The accuracy is {(correct_ones/Y.shape[0])*100} % \n")

    #Show values and function
    utils.plot_data(X, Y, "y_=1", "y_=0", 'green', 'blue')
    utils.plot_decision_boundary(w, b, X_stack, Y)
    
    plt.legend()
    plt.show()

def public_Test():
    test.sigmoid_test(lr.sigmoid)
    test.compute_cost_test(lr.compute_cost)
    test.compute_cost_reg_test(lr.compute_cost_reg)
    test.compute_gradient_reg_test(lr.compute_gradient_reg)

def readData(file):
    data = np.loadtxt("./data/" + str(file), delimiter=',', skiprows=1)
    X_train = data[:, :2]
    y_train = data[:, 2]

    return X_train, y_train

def main():
    # our_test_A()
    our_test_B()
    # public_Test()
    # utils.plot_data(X, Y)

if __name__ == '__main__':
    main()