import public_tests as test
import numpy as np
import matplotlib.pyplot as plt
import logistic_reg as lr

def our_test():
    x = 0

def public_Test():
    # test.sigmoid_test(lr.sigmoid)
    # test.compute_cost_test(lr.compute_cost)
    test.compute_gradient_test(lr.compute_gradient)

def main():
    public_Test()

if __name__ == '__main__':
    main()