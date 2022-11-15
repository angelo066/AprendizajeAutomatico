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

def sobreAjuste(degree, XTrain, y_train, XTest, y_test):
    poly = PolynomialFeatures(degree, include_bias=False)
    xTrainMapped = poly.fit_transform(XTrain) # entrena X

    scaler = StandardScaler() # normalizacion
    XtrainMappedScaled = scaler.fit_transform(xTrainMapped)

    linear_model = LinearRegression()
    linear_model.fit(XtrainMappedScaled, y_train)

    #Datos de entrenamiento
    train, yTrainPredicted = calcError(XTrain, y_train, poly, scaler, linear_model)
    print("Train:" + str(train))
    test , yTestPredicted = calcError(XTest, y_test, poly, scaler, linear_model)
    print("Test:" + str(test))

    return yTrainPredicted , yTestPredicted


def main():
    x_train, y_train, x_ideal, y_ideal = gen_data(64)
    plt.figure()
    plt.plot(x_ideal, y_ideal, c = 'red', label = 'y_ideal', linestyle = '--')
    plt.scatter(x_train, y_train, c = 'blue', label = 'train', marker= 'o')

    X_trainData, X_testData, Y_trainData, Y_testData = train_test_split(x_train, y_train, 
                                                        test_size=0.33, random_state= 1) 
    X_trainData_matrix = X_trainData[:, None]
    X_testData_matrix = X_testData[:, None]

    yTrainPredict , yTestPredict = sobreAjuste(15, X_trainData_matrix, Y_trainData, X_testData_matrix, Y_testData)
    
    #Juntamos y ordenamos
    # np.concatenate((X_trainData, X_testData)
    XData_ = np.sort(X_trainData ,axis=None)
    YData = np.sort(yTrainPredict ,axis=None)

    plt.plot(XData_, YData, c = 'red', label = 'predicted')
    plt.legend()
    plt.show()
    # show_samples()

if __name__ == '__main__':
    main()