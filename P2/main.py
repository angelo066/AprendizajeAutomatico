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

    X_train_norm , mu, sigma = ml.zscore_normalize_features(X_train)

    ml.compute_cost(X_train_norm, y_train, w_init, b_init)

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
    # our_test()
    public_Test()

main()