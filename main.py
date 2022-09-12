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
    x = 0 
    y = 0
     
    integral = (Ndebajo / num_puntos) * (b - a) * M
    
    return integral

#Función que utilizamos para la integral
def cuadrado(x):
    return x * x

def main() :
    print(cuadrado(2), "\n")


main()