from array import array
from random import uniform
import numpy as np
import scipy
from scipy import integrate

def integraVectores(fun, a, b, num_puntos = 10000):
    M = b ** 2

    vectorX = np.random.uniform(a,b,[1,num_puntos])

    vectorY = np.random.uniform(0,M,[1,num_puntos])

    comparatoria = np.array(fun(vectorX))

    Ndebajo = np.sum(vectorY < comparatoria)

    return (Ndebajo / num_puntos) * (b - a) * M



def intengra_mc(fun, a, b, num_puntos = 10000):
    integral = 0
    Ndebajo = 0 
    M = b ** 2
    
    #Esto lo hariamos en caso de que la función no fuese el caudrado para buscar la imagen de la misma, al ser el cuadrado podemos estar seguros
    #de que la imagen es b al cuadrado
    #M = 0 
    # max = 0
    # for i in range(a,b):
        
    #     cuad = cuadrado(i)
        
    #     if(cuad > max) : max = cuad
        
    # #buscamos la imagen de la funcion
    # M = max; 
    
    #puntos generados aleatoriamente 
    for i in range (0,num_puntos):
        x = uniform(a,b)
        y = uniform(0, M)

        if(fun(x) > y) : Ndebajo += 1 
     
    integral = (Ndebajo / num_puntos) * (b - a) * M
    
    return integral

#Función que utilizamos para la integral
def cuadrado(x):
    return x * x

def main() :
    print("Nuestra solucion:" ,intengra_mc(cuadrado, 1, 5), "\n")
    print("Solucion de vectores", integraVectores(cuadrado,1,5), "\n")
    print("Solucion real:", scipy.integrate.quad(cuadrado, 1, 5), "\n")




main()