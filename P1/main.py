import utils
import linear_reg as ln
import public_tests as test
import matplotlib.pyplot as plt
import numpy as np

def linear_regression(w, b, x):
    return w * x + b

def our_test():
    X, Y =utils.load_data()

    initial_w = 2.0
    initial_b = 3.0

    iterations = 1500
    alpha = 0.01

    w , b, history = ln.gradient_descent(X, Y, initial_w, initial_b, ln.compute_cost, ln.compute_gradient, alpha , iterations)

    print("w, b found by gradient descent:", w, b)

    range_MAX = np.max(X)
    range_MIN = np.min(X)
    X_fun = np.linspace(range_MIN, range_MAX, 256)
    Y_fun = np.array(linear_regression(w, b, X_fun)) 

    #Represent Data
    plt.figure()
    plt.plot(X_fun, Y_fun, c = 'blue', label = 'Regression : y = ' + "{:.2f}".format(w) + " * " + "x + " + "{:.2f}".format(b))
    plt.scatter(X, Y, c = 'red', label = 'Data', marker= 'x')
    plt.legend()
    # plt.show()
    plt.savefig('linearRegression_prediction.pdf')

def execute_tests():
    test.compute_cost_test(ln.compute_cost)
    test.compute_gradient_test(ln.compute_gradient)

def main():
    execute_tests()
    our_test()

if __name__ == '__main__':
    main()