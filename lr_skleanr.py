# Linear Regression using scikit learn


import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data_matrix = np.loadtxt('ex1data2.txt',delimiter = ",",unpack= True)
#print(data_matrix)

X = np.transpose(np.array(data_matrix[:-1]))
y = np.transpose(np.array(data_matrix[-1:]))

#X = np.insert(X,0,1,axis=1)

regr = LinearRegression()
regr.fit(X,y)

print("Coefficients: ",regr.coef_)

print(regr.predict([1650,3]))
