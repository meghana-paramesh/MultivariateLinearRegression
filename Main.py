from MultiVariateLinearRegression import LinearRegression
import pandas as pd
from NormalEquation import normal_equation
import numpy as np

if __name__ == "__main__":

    linreg = LinearRegression(learning_rate=0.01, epochs=10000)

    # We can ignore the header of the dataframe using header=None, hence the first line is not skipped
    df = pd.read_csv("data1.txt", sep=",", header=None)
    X_train = df.iloc[:, 0:2]
    normalized_X_train = (X_train - X_train.mean()) / X_train.std()
    normalized_X_train = normalized_X_train.to_numpy().reshape(X_train.shape[0], X_train.shape[1])
    y_train = df.iloc[:, 2]
    y_train = y_train.to_numpy().reshape(len(normalized_X_train), 1)
    file = open("output.txt", "w")
    y_predicted = linreg.fit(normalized_X_train, y_train, "l2_norm_based")
    file.write("=======================================\n")
    file.write("Gradient Descent Method\n")
    file.write("=======================================\n")
    print("=======================================")
    print("Gradient Descent Method")
    print("=======================================")
    file.write("Values of theta using Gradient Descent Method\n")
    print("Values of theta using Gradient Descent Method")

    file.write("theta_0: "+str(linreg.theta_0)+"\n")
    file.write("theta_1: "+ str(linreg.theta_1[0])+"\n")
    file.write("theta_2: "+str(linreg.theta_1[1])+"\n")
    print("theta_0: ", linreg.theta_0)
    print("theta_1: ", linreg.theta_1[0])
    print("theta_2: ", linreg.theta_1[1])
    X_test = [1650, 3]
    normalized_X_test = (X_test - X_train.mean()) / X_train.std()
    predictions = linreg.predict(normalized_X_test)
    file.write("price prediction for a 1650-square-foot house with 3 bedrooms using gradient descent: "+str(predictions)+"\n")
    print("price prediction for a 1650-square-foot house with 3 bedrooms using gradient descent: ", predictions)

    file.write("=======================================\n")
    file.write("Normal Equation Method\n")
    file.write("=======================================\n")
    print("=======================================")
    print("Normal Equation Method")
    print("=======================================")
    # normalized equation method
    X_train = X_train.to_numpy().reshape(X_train.shape[0], X_train.shape[1])
    ones = np.ones((X_train.shape[0], 1))
    X_train = np.hstack((ones, X_train))
    file.write("Values of theta using Normal equation in the format [[theta_0][theta_1][theta_2]]\n")
    print("Values of theta using Normal equation in the format [[theta_0][theta_1][theta_2]]")
    normal_equation_thetas = normal_equation(X_train, y_train)
    file.write(str(normal_equation_thetas)+"\n")
    print(normal_equation_thetas)
    X_test = [1, 1650, 3]
    predictions = np.dot(X_test, normal_equation_thetas)
    file.write("price prediction for a 1650-square-foot house with 3 bedrooms using normal equations: "+str(predictions)+"\n")
    print("price prediction for a 1650-square-foot house with 3 bedrooms using normal equations: ",
          predictions)
