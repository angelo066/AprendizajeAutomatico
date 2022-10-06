import public_tests as test
import multi_linear_reg as ml
import numpy as np
import matplotlib.pyplot as plt

def our_test():
    data = np.loadtxt("./data/houses.txt", delimiter=',', skiprows=1)
    X_train = data[:, :4]
    y_train = data[:, 4]

    b_init = 785.1811367994083
    w_init = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])

    X_train_norm, mu, sigma = ml.zscore_normalize_features(X_train)

    iterations = 1500
    alpha = 0.01
    # #TRAINING
    w , b, history = ml.gradient_descent(X_train_norm, y_train, w_init, b_init, ml.compute_cost, 
                                         ml.compute_gradient, alpha, iterations)
    
    X = np.array([1200, 3, 1, 40])
    X = (X - mu) / sigma
    test = np.sum((X @ w)) + b
    print(test)

def public_Test():
    test.compute_cost_test(ml.compute_cost)
    test.compute_gradient_test(ml.compute_gradient)

def main():
    public_Test()
    our_test()

if __name__ == '__main__':
    main()