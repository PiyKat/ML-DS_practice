# ex1 - Linear regression with one variable

import numpy as np
import matplotlib.pyplot as plt
import math

file_data = np.loadtxt('ex1data1.txt',delimiter=",",unpack=True)

# extracting input and output from data

X = np.transpose(np.array(file_data[:-1]))
print("Input Matrix :\n")
#print(X)
y = np.transpose(np.array(file_data[-1:]))
print("Output Matrix: \n")
#print(y.shape)
m = len(X)


# Adding x0 value to the input matrix(used for theta calculation

X = np.insert(X,0,1,axis=1)  # Note to self - axis=1 means row
#print(X)

plt.xlabel("City Population(In 10000)")
plt.ylabel("Housing Prices in $100000")
plt.plot(X[:,1],y[:,0],'bo',markersize = 3)
plt.show()

# function to compute hypothesised value

def H_Theta(X,initTheta):
    return np.dot(X,initTheta)



def ComputeCost(X,y,init_Theta):
    return float((1./(2*m)) * np.dot((H_Theta(X,init_Theta)-y).T,(H_Theta(X,init_Theta)-y)))
##    calculated_cost = (1./2*m)*(np.dot(np.transpose((H_Theta(X,init_Theta)-y)),(H_Theta(X,init_Theta)-y)))
##    return calculated_cost


def LinearRegression(X,y,init_Theta,iterations=1500):    
    alpha = 0.01
    theta = init_Theta
    cost_history = []
    for i in range(iterations):  
        tempTheta = theta
        calculated_cost = ComputeCost(X,y,tempTheta)   
        #print(calculated_cost)
        
        cost_history.append(calculated_cost)

        for j in range(len(tempTheta)):
            tempTheta[j] = theta[j] - (alpha/m)*np.sum((H_Theta(X,init_Theta) - y)* np.array(X[:,j]).reshape(m,1))
            
        theta = tempTheta
        #print(tempTheta)
    return cost_history, theta
    #print(cost_history)
    print(theta)

initial_theta = np.zeros((X.shape[1],1))
cost,theta = LinearRegression(X,y,initial_theta,1500)
print(cost[:50])
print(theta)

def plot_line(x):
    return theta[0] + theta[1]*x


plt.plot(X[:,1],y[:,0],'bo')
plt.plot(X[:,1],plot_line(X[:,1]))
plt.show()

def plotConvergence(cost):
    #plt.figure(figsize=(10,6))
    plt.plot(range(len(cost)),cost,'bo')
    plt.grid(True)
    plt.title("Convergence of Cost Function")
    plt.xlabel("Iteration number")
    plt.ylabel("Cost function")
    dummy = plt.xlim([-0.05*1500,1.05*1500])
    plt.show()

plotConvergence(cost)
dummy = plt.ylim([4,7])
        
        


