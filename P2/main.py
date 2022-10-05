import public_tests as test
import multi_linear_reg as ml
import numpy as np
import matplotlib.pyplot as plt

def our_test():
    data = np.loadtxt("./data/houses.txt", delimiter=',', skiprows=1)
    X_train = data[:, :4]
    y_train = data[:, 4]
    # X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
    # y_train = np.array([460, 232, 178])


    b_init = 785.1811367994083
    w_init = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])
    # b_init = 0.0
    # w_init = np.array([0, 0, 0, 0])


    X_train_norm = ml.zscore_normalize_features(X_train)[0]
    iterations = 1500
    alpha = 0.01
    # #TRAINING
    w , b, history = ml.gradient_descent(X_train_norm, y_train, w_init, b_init, ml.compute_cost, 
                                         ml.compute_gradient, alpha, iterations)

    X = np.array([1200, 3, 1, 40])
    X = ml.zscore_normalize_features(X)[0]
    test = np.sum((X @ w) + b)
    print(test)

    # X_features = ['size(sqft)', 'bedrooms', 'floors', 'age']
    # fig, ax = plt.subplots(1, 4, figsize=(25, 5), sharey=True)
    # for i in range(len(ax)):
    #     ax[i].scatter(X_train[:, i], y_train)
    #     ax[i].set_xlabel(X_features[i])
    #     ax[0].set_ylabel("Price (1000's)")
    # plt.show()

def public_Test():
    test.compute_cost_test(ml.compute_cost)
    test.compute_gradient_test(ml.compute_gradient)

def main():
    # public_Test()
    our_test()

main()