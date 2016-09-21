# Linear regression using normal equation for multiple features

import numpy as np
import matplotlib.pyplot as plt

data_matrix = np.loadtxt('ex1data2.txt',delimiter = ",",unpack= True)
#print(data_matrix)

X = np.transpose(np.array(data_matrix[:-1]))
y = np.transpose(np.array(data_matrix[-1:]))

m = len(X)
n = X.shape[1] + 1


X = np.insert(X,0,1,axis=1)
#print(X)

def normalizeData(X):
    mean = []
    data_range = []
    mean.append(np.mean(X[:,1]))
    mean.append(np.mean(X[:,2]))
    data_range = np.ptp(X,axis=0)[-2:]
    #print(mean,data_range)
    for i in range(len(X)):
        X[:,1][i]  = (X[:,1][i] - float(mean[0]))/float(data_range[0])
        X[:,2][i] = (X[:,2][i] - float(mean[1]))/float(data_range[1])

    return X

def Normal_Equation(X,y):
    #return np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,y))
    return np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,y))


theta = Normal_Equation(X,y)
print(theta)
