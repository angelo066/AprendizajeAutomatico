import utils
import linear_reg as ln
import public_tests as test
import matplotlib.pyplot as plt
import numpy as np
def main():
    X, Y =utils.load_data()

    # X = np.array([2, 4, 6, 8])
    # Y = np.array([7, 11, 15, 19])

    cost = ln.compute_cost(X, Y, 2.0, 3.0)

    print (cost)

    plt.figure()
    plt.scatter(X, Y, c = 'red', label = 'Data', marker= 'x')
    plt.legend()
    plt.show()
    # plt.savefig('timesComparation.png')
    # print (X,"\n", Y)

main()