import numpy as np
import matplotlib.pyplot as plt
import utils as utils
from scipy.io import loadmat


import multi_class as mC
import logistic_reg as lr
import public_tests as test
# def showData():
#     w , b , X, Y = our_test_B()

#     correct_ones = 0
#     for i in range(Y.shape[0]):
#         if(Y[i] == lr.predict(X[i], w,b)):
#             correct_ones +=1
    
#     accuracy =(f"The accuracy is {(correct_ones/Y.shape[0])*100} %")
#     print(accuracy)

#     plt.title(accuracy)
#     #Show values and function
#     # utils.plot_data(X, Y, "y_=1", "y_=0", 'green', 'blue')
#     utils.plot_decision_boundary(w, b, X, Y)
#     plt.legend()
#     plt.savefig('partB.pdf')
#     # plt.show()

# def our_test_A():
#     #read data
#     X , Y = readData("ex2data1.txt")
#     #initial values
#     b_init = -8
#     w_init = np.array([0.0, 0.0])
#     iterations = 1000
#     alpha = 0.001
#     #TRAINING
#     w , b, history = lr.gradient_descent(X, Y, w_init, b_init,lr.compute_cost, lr.compute_gradient, alpha , iterations)
#     #Predict Values
#     return w, b, X, Y
    

# def our_test_B():
#     #read data
#     X , Y = readData("ex2data2.txt")
#     #initial values
#     b_init = 1
#     iterations = 10000
#     alpha = 0.01

#     #TRAINING
#     #X[:, 0] -z> : todas las filas de la columna 0

#     X_stack = utils.map_feature(X[:,0],X[:,1])
#     w_init = np.zeros(X_stack.shape[1])
 
#     w , b, history = lr.gradient_descent(X_stack, Y, w_init, b_init,lr.compute_cost_reg, lr.compute_gradient_reg, alpha , iterations)

#     return w, b, X_stack, Y

def public_Test():
    test.sigmoid_test(lr.sigmoid)
    test.compute_cost_test(lr.compute_cost)
    test.compute_gradient_test(lr.compute_gradient)
    test.compute_cost_reg_test(lr.compute_cost_reg)
    test.compute_gradient_reg_test(lr.compute_gradient_reg)

def show_samples():
    X , Y = readData("ex3data1.mat")
    
    rand_indices = np.random.choice(X.shape[0], 100, replace=False)
    utils.displayData(X[rand_indices, :])
    plt.show()

def our_test():
    X , Y = readData("ex3data1.mat")

    n_label = 10
    alpha = 0.1
    all_theta = mC.oneVsAll(X, Y, n_label, alpha)
    
    p = mC.predictOneVsAll(all_theta, X)
    
    acertados = 0
    for i in range(p.shape[0]):
        if(Y[i] == p[i]):
            acertados += 1
            
    print((acertados / Y.shape[0]) * 100)
    


def readData(file):
    data = loadmat('data/ex3data1.mat', squeeze_me=True)

    X_train = data['X']
    y_train = data['y']

    return X_train, y_train

def main():
    # show_samples()
    our_test()    
    # public_Test()


if __name__ == '__main__':
    main()