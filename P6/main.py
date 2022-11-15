import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.interpolate import make_interp_spline

def gen_data(m, seed=1, scale=0.7):
    """ generate a data set based on a x^2 with added noise """
    c = 0
    x_train = np.linspace(0, 49, m)
    np.random.seed(seed)
    y_ideal = x_train**2 + c
    y_train = y_ideal + scale * y_ideal*(np.random.sample((m,))-0.5)
    x_ideal = x_train
    return x_train, y_train, x_ideal, y_ideal


def predict(X, polynomialInstance, scaleInstance, linearRegInstance):
    xMapped = polynomialInstance.transform(X)
    XMappedScaled = scaleInstance.transform(xMapped)
    return linearRegInstance.predict(XMappedScaled)

def calcError(X, Y, poly, scaler, linearModel):
    m = Y.shape[0]

    error = 0
    yhat =  predict(X, poly, scaler, linearModel)
    for i in range(m):
        error += ((yhat[i] - Y[i]) ** 2) 

    return error / (2 * m ), yhat 

def trainPolinomic(degree, XTrain, y_train):
    poly = PolynomialFeatures(degree, include_bias=False)
    xTrainMapped = poly.fit_transform(XTrain) # entrena X

    scaler = StandardScaler() # normalizacion
    XtrainMappedScaled = scaler.fit_transform(xTrainMapped)

    linear_model = LinearRegression()
    linear_model.fit(XtrainMappedScaled, y_train)

    return poly, scaler, linear_model

def overFit(X, Y):
    #train -> 67% 
    #test -> 33%
    X_trainData, X_testData, Y_trainData, Y_testData = train_test_split(X, Y, 
                                                        test_size=0.33, random_state= 1) 
    X_trainData_matrix = X_trainData[:, None]
    X_testData_matrix = X_testData[:, None]

    poly, scaler, linear_model = trainPolinomic(15, X_trainData_matrix, Y_trainData)

    #Calculamos errores    
    train, yTrainPredict = calcError(X_trainData_matrix, Y_trainData, poly, scaler, linear_model)
    print("Train:" + str(train))
    test , yTestPredict = calcError(X_testData_matrix, Y_testData, poly, scaler, linear_model)
    print("Test:" + str(test))

    #Todos los ejemplos
    # XData_ = np.sort(np.concatenate((X_trainData, X_testData) ,axis=None)) 
    # YData = np.sort(np.concatenate((yTrainPredict, yTestPredict) ,axis=None))
    
    #ordenamos
    #Datos entrenamiento
    XData_ = np.sort(X_trainData,axis=None)
    YData = np.sort(yTrainPredict ,axis=None)
    plt.plot(XData_, YData, c = 'red', label = 'predicted')


def optimumDegree(X, Y):
    #train  -> 60%
    #validacion  -> 20%
    #test  -> 20%
    X_trainData, X_testData, Y_trainData, Y_testData = train_test_split(X, Y, 
                                                        test_size=0.4, random_state= 1)

    X_validateData, X_testData, Y_validateData, Y_testData = train_test_split(X_testData, Y_testData, 
                                                        test_size=0.5, random_state= 1)

    X_trainData_matrix = X_trainData[:, None]
    X_testData_matrix = X_testData[:, None]
    X_validateData_matrix = X_validateData[:, None]

    optimumDegree = None
    minErrorValidate = float('inf')
    for i in range(1,11):
        poly, scaler, linear_model = trainPolinomic(i, X_trainData_matrix, Y_trainData)
        
        error = calcError(X_validateData_matrix, Y_validateData, poly, scaler, linear_model)[0]
        # print(f"Validate {i} :" + str(error))
        if(error < minErrorValidate):
            minErrorValidate = error
            optimumDegree = i

    print(f"Optimum degree is {optimumDegree}")

    poly, scaler, linear_model = trainPolinomic(optimumDegree, X_trainData_matrix, Y_trainData)
        
    error, yValidatePredict = calcError(X_validateData_matrix, Y_validateData, poly, scaler, linear_model)

    XData_ = np.sort(X_validateData,axis=None)
    YData = np.sort(yValidatePredict ,axis=None)
    plt.plot(XData_, YData, c = 'blue', label = 'predicted')


def main():
    x_train, y_train, x_ideal, y_ideal = gen_data(64)
    plt.figure()
    plt.plot(x_ideal, y_ideal, c = 'red', label = 'y_ideal', linestyle = '--')
    plt.scatter(x_train, y_train, c = 'blue', label = 'train', marker= 'o')

    # overFit(x_train, y_train)
    optimumDegree(x_train, y_train)

    plt.legend()
    plt.show()
    # show_samples()

if __name__ == '__main__':
    main()