from cmath import exp
import math
import numpy as np
import matplotlib.pyplot as plt
from random import uniform
import scipy
from scipy import integrate
import time

def cuadrado(x):
    return x * x

def exponential(x):
    return -0.5 * ((x-4)**2) + 5

def logBaseTen(x):
    return math.log10(x)

def calcMaxHeight(func, a, b, num_samples = 300):
    maxHeight = 0
    interval = (b - a) / num_samples
    acc = interval
    for i in range(num_samples):
        image = func(a + acc)
        if(image > maxHeight):
            maxHeight = image
        acc += interval
    return maxHeight

def intengra_mc(fun, a, b, num_puntos = 10000):
    tic = time.process_time()
    integral = 0
    Ndebajo = 0 
    M = calcMaxHeight(fun, a, b)
        
    #puntos generados aleatoriamente 
    for i in range (0,num_puntos):
        x = uniform(a,b)
        y = uniform(0, M)
        if(fun(x) > y) : Ndebajo += 1 
     
    integral = (Ndebajo / num_puntos) * (b - a) * M
    
    # return integral
    toc = time.process_time()
    return 1000* (toc - tic) , integral

def integraVectores(fun, a, b, num_puntos = 10000):
    tic = time.process_time()
    M = b ** 2
    vectorX = np.random.uniform(a,b,[1,num_puntos])

    vectorY = np.random.uniform(0,M,[1,num_puntos])

    comparatoria = np.array(fun(vectorX))

    Ndebajo = np.sum(vectorY < comparatoria)

    toc = time.process_time()

    return 1000* (toc - tic) , (Ndebajo / num_puntos) * (b - a) * M

def GenerateTimeGraph():
    sizes = np.linspace(100, 1000000, 20).round().astype(int)

    time_vector = []
    time_point = []

    for size in sizes:
        time_vector += [integraVectores(cuadrado,1,5, size)[0]]
        time_point += [intengra_mc(cuadrado,1,5, size)[0]]
    
    plt.figure()
    plt.scatter(sizes, time_vector, c = 'red', label = 'Vector')
    plt.scatter(sizes, time_point, c = 'blue', label = 'Points')
    plt.legend()
    plt.show()
    # plt.savefig('sinbad.png')

def main() :
    print("Nuestra solucion:" ,intengra_mc(cuadrado, 1, 5)[1], "\n")
    print("Solucion de vectores", integraVectores(cuadrado,1,5)[1], "\n")
    print("Solucion real:", scipy.integrate.quad(cuadrado, 1, 5), "\n")
    GenerateTimeGraph()

main()