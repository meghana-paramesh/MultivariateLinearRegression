import numpy as np

def normal_equation(X,y):
    temp = np.linalg.inv((X.T).dot(X))
    return temp.dot(X.T).dot(y)