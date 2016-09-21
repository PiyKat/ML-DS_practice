# Linear regression for multiple features

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

def H(X,initTheta):
    X1 = normalizeData(X)
    cost_function = float((1./2*m)*(np.dot((np.dot(X1,initTheta)-y).T,(np.dot(X1,initTheta)-y))))
    return(cost_function)

alpha = 0.01
iterations = 400


def LinReg_Mult(X,y,initTheta):
    X1 = normalizeData(X)
    theta = initTheta
    cost_data = []
    for i in range(iterations):
        tempTheta = theta
        cost = H(X,tempTheta)
        cost_data.append(cost)
        for j in range(len(tempTheta)):
            tempTheta[j] = tempTheta[j] - (alpha/m)* np.dot((np.dot(X1,tempTheta)-y).T,np.array(X1[:,j]).reshape(m,1))
        theta=tempTheta
    return theta,cost_data

theta,cost = LinReg_Mult(X,y,initTheta = np.zeros((n,1)))
print(theta)
print(cost)


def plotConvergence(cost):
    #plt.figure(figsize=(10,6))
    plt.plot(range(len(cost)),cost,'bo')
    plt.grid(True)
    plt.title("Convergence of Cost Function")
    plt.xlabel("Iteration number")
    plt.ylabel("Cost function")
    dummy = plt.xlim([-0.05*400,1.05*400])
    plt.show()

plotConvergence(cost)

                    
                
                
   




