import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model.logistic import _logistic_loss
import matplotlib.pyplot as plt

data = np.loadtxt('ex2data1.txt',delimiter=",",unpack = True)

X = np.array(np.transpose(data[:-1]))
y = np.array(np.transpose(data[-1:]))

#print(X)
#print(y)

print(y.shape)

clf = LogisticRegression()
clf.fit(X,y)

print("Coefficients : ",clf.coef_)
