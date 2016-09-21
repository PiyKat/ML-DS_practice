import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt

cols = np.loadtxt('ex2data1.txt',delimiter=",",usecols =(0,1,2),unpack = True)
#print(cols)

X = np.array(np.transpose(cols[:-1]))  # If written in last, the last row dissapears
y = np.array(np.transpose(cols[-1:]))  # If written in first, the last row appears
#print(X)
m = len(X)
X = np.insert(X,0,1,axis=1)
n = X.shape[1]
#print(y)

# Using feature scaling on data
def VisualizeData(X,y):
    positive_examples = np.array([X[i] for i in range(len(X)) if y[i] == 1])
    negative_examples = np.array([X[i] for i in range(len(X)) if y[i] == 0])
    plt.plot(positive_examples[:,1],positive_examples[:,2],'b+',label="Admitted")
    plt.plot(negative_examples[:,1],negative_examples[:,2],'ro',label="Not Admitted")
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    



def FeatureScaling(X):
    mean = []
    data_range = []
    X1 = np.zeros((len(X),X.shape[1]))
    mean.append(np.mean(X[:,1]))
    mean.append(np.mean(X[:,2]))
    data_range = np.ptp(X,axis=0)[-2:]
    #print(mean)
    print(data_range)
    for i in range(len(X)):
        X1[:,0][i] = (X[:,0][i] - mean[0])/data_range[0]
        X1[:,1][i] = (X[:,1][i] - mean[1])/data_range[1]
    return X1

def H(initTheta,X):
    # X1 = FeatureScaling(X) 
    hypothesis = expit(np.dot(X,initTheta))
    return hypothesis

initTheta = np.zeros((X.shape[1],1))
alpha = 0.0012
iterations = 200

def CostCompute(initTheta,X,y):
    X1 = FeatureScaling(X)
    cost_record = []
    theta = initTheta
    for x in range(iterations):
        tempTheta = theta
        
        cost = (-1./m)*(np.dot(y,np.log(H(tempTheta,X)).T) + np.dot((1-y),np.log(1-H(tempTheta,X)).T))
        cost_record.append(np.sum(cost))
        for i in range(X.shape[1]):
            
            tempTheta[i] = tempTheta[i] - (alpha/m)*(np.dot((H(tempTheta,X)-y).T,X[:,i]))
        theta = tempTheta
        
    
    return cost_record,theta

cost_data,theta = CostCompute(initTheta,X,y)
print(cost_data)
print(theta)

print (H(theta,np.array([1, 45.,85.])))

boundary_xs = np.array([np.min(X[:,1]), np.max(X[:,1])])
print(boundary_xs)
##boundary_ys = (-1./theta[2])*(theta[0] + theta[1]*boundary_xs)
##VisualizeData(X,y)
##plt.plot(boundary_xs,boundary_ys,'b-',label='Decision Boundary')
##plt.show()
##    
##plt.plot(range(len(cost_data)),cost_data,'bo')
##plt.title("Convergence of Cost Function")
##plt.xlabel("Iteration number")
##plt.ylabel("Cost function")         
##plt.xlim([-0.05*iterations,1.05*iterations])
##plt.show()


