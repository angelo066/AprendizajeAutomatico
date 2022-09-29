from cmath import exp
import math
import numpy as np
import matplotlib.pyplot as plt
from random import uniform
import scipy
from scipy import integrate
import time

#Funciones que podemos utilizar
def cuadrado(x):
    return x * x

def exponential(x):
    return -0.5 * ((x-4)**2) + 5

def logBaseTen(x):
    return math.log10(x)

#Funcion para calcular la altura maxima de nuestra funcion en concreto
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

#Funcion en la que integramos a traves del método de monte carlo SIN vectores
def intengra_mc(fun, a, b, num_puntos = 20000):
    tic = time.process_time()
    integral = 0
    Ndebajo = 0 
    M = calcMaxHeight(fun, a, b)
        
    #puntos generados aleatoriamente 
    for i in range (0,num_puntos):
        x = uniform(a,b)
        y = uniform(0, M)
        if(fun(x) > y) : Ndebajo += 1 #si está debajo de la gráfica, sumamos 1
     
    integral = (Ndebajo / num_puntos) * (b - a) * M
    
    toc = time.process_time()
    return 1000* (toc - tic) , integral

#Funcion en la que integramos a traves del método de monte carlo UTILIZANDO vectores
def integraVectores(fun, a, b, num_puntos = 20000):
    tic = time.process_time()
    M = b ** 2
    vectorX = np.random.uniform(a,b,[1,num_puntos])     #Coordenadas X de los puntos
    vectorY = np.random.uniform(0,M,[1,num_puntos])     #Coordenadas Y de los puntos

    comparatoria = np.array(fun(vectorX))               #Calculamos el punto Y correspondiente para cada punto X
    Ndebajo = np.sum(vectorY < comparatoria)            #Calculamos cuantos puntos han caido debajo de la gráfica

    toc = time.process_time()

    return 1000* (toc - tic) , (Ndebajo / num_puntos) * (b - a) * M

#Cálculo y dibujado del tiempo de la funcion
def GenerateTimeGraph():
    sizes = np.linspace(100, 2000000, 20).round().astype(int)

    time_vector = []
    time_point = []

    for size in sizes:
        time_vector += [integraVectores(cuadrado,1,5, size)[0]]
        time_point += [intengra_mc(cuadrado,1,5, size)[0]]
    
    plt.figure()
    plt.scatter(sizes, time_vector, c = 'red', label = 'Vector')
    plt.scatter(sizes, time_point, c = 'blue', label = 'Points')
    plt.legend()
    # plt.show()
    plt.savefig('timesComparation.png')

def main() :
    print("Nuestra solucion:" ,intengra_mc(cuadrado, 1, 5)[1], "\n")
    print("Solución de vectores", integraVectores(cuadrado,1,5)[1], "\n")
    print("Solución real:", scipy.integrate.quad(cuadrado, 1, 5), "\n")
    GenerateTimeGraph()

main()