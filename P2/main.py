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
    target_value = 318.85441519816953
    assert np.isclose(test, target_value,
                      rtol=1e-4), f"Case 1: prediction is wrong: {test} != {target_value}"
    print("\033[92mTest prediction passed!")

    Y_prediction = (X_train_norm @ w) + b

    X_features = ['size(sqft)', 'bedrooms', 'floors', 'age']
    fig, ax = plt.subplots(1, 4, figsize=(25, 5), sharey=True)
    plt.title('Target versus prediction using z-score normalized model.')
    for i in range(len(ax)):
        ax[i].scatter(X_train_norm[:, i], y_train, label = 'target')
        ax[i].scatter(X_train_norm[:, i], Y_prediction, color = 'orange', label = 'prediction')
        ax[i].set_xlabel(X_features[i])
        
    plt.legend()
    ax[0].set_ylabel("Price (1000's)")
    

    plt.show()
    # plt.savefig('linearRegression_prediction.pdf')


def public_Test():
    test.compute_cost_test(ml.compute_cost)
    test.compute_gradient_test(ml.compute_gradient)

def main():
    public_Test()
    our_test()

if __name__ == '__main__':
    main()