def intengra_mc(fun, a, b, num_puntos = 10000):
    integral = 0
    Ndebajo = 0; 
    M = 0; 
     
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