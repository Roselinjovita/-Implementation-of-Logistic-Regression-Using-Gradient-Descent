# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm


1.Use the standard libraries in python for finding linear regression.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Predict the values of array.

5.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

6.Obtain the graph.
 

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: S.ROSELIN MARY JOVITA
RegisterNumber:  212222230122
*/

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data= np.loadtxt("ex2data1.txt", delimiter=",")
X = data[:,[0,1]]
y = data[:,2]


X[:5]

y[:5]

plt.figure()
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label="Admitted")
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label=" Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()


def sigmoid(z):
  return 1/(1+np.exp(-z))

  plt.plot()
X_plot =np.linspace(-10, 10, 100)
plt.plot(X_plot, sigmoid(X_plot))
plt.show()


def costFunction(theta,X,y):
  h = sigmoid(np.dot(X, theta))
  J = -(np.dot(y, np.log(h)) + np.dot(1 - y, np.log(1-h))) / X.shape[0]
  grad = np.dot(X.T, h-y) / X.shape[0]
  return J,grad

X_train = np.hstack((np.ones((X.shape[0], 1)), X))
theta = np.array([0, 0, 0])
J, grad = costFunction(theta, X_train, y)
print(J)
print(grad)


X_train = np.hstack((np.ones((X.shape[0], 1)), X))
theta = np.array([-24, 0.2, 0.2  ])
J, grad = costFunction(theta, X_train, y)
print(J)
print(grad)


def cost(theta, X, y):
  h= sigmoid(np.dot(X, theta))
  J = - (np.dot(y, np.log(h)) + np.dot(1 - y, np.log(1 - h))) / X.shape[0]
  return J

  def gradient(theta, X, y):
  h = sigmoid(np.dot(X, theta))
  grad = np.dot(X.T, h - y) / X.shape[0]
  return grad

  X_train = np.hstack((np.ones((X.shape[0], 1)), X))
theta = np.array([0, 0, 0])
res = optimize.minimize(fun=cost, x0=theta, args=(X_train, y),
                        method='Newton-CG', jac=gradient)
print(res.fun)
print(res.x)


def plotDecisionBoundary(theta, X, y):
  x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  y_min, y_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                       np.arange(y_min, y_max, 0.1))
  X_plot = np.c_[xx.ravel(), yy.ravel()]
  X_plot = np.hstack((np.ones((X_plot.shape[0], 1)), X_plot))
  y_plot = np.dot(X_plot, theta).reshape(xx.shape)

  plt.figure()
  plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label="Admitted")
  plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label="Not admitted")
  plt.contour(xx, yy, y_plot, levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()



  plotDecisionBoundary(res.x, X, y)


  prob = sigmoid(np.dot(np.array([1, 45, 85]), res.x))
print(prob)


def predict(theta, X):
  X_train = np.hstack((np.ones((X.shape[0], 1)), X))
  prob = sigmoid(np.dot(X_train, theta))
  return (prob >= 0.5).astype(int)


  np.mean(predict(res.x, X) == y)
```

## Output:

![Screenshot 2023-09-25 174631](https://github.com/Roselinjovita/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119104296/c0e7ccf6-596f-426c-9ec6-bb6c44db5279)

![Screenshot 2023-09-25 174721](https://github.com/Roselinjovita/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119104296/18c2caf9-6cb8-425e-af9d-2cc5793aa16c)



![Screenshot 2023-09-25 174802](https://github.com/Roselinjovita/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119104296/fd6cb900-697f-4525-8c84-19890bc8af11)

![Screenshot 2023-09-25 174915](https://github.com/Roselinjovita/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119104296/5a24f549-2334-4e2b-bfdb-ba3f42db8bb9)



![Screenshot 2023-09-25 174943](https://github.com/Roselinjovita/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119104296/2fc97b75-229a-4531-bd54-1db63c0d96a9)




![Screenshot 2023-09-25 175024](https://github.com/Roselinjovita/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119104296/49c4267f-7764-44d8-8e59-e384d19405d9)


![Screenshot 2023-09-25 175052](https://github.com/Roselinjovita/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119104296/9b6c0ec7-6ba9-4f37-aaf3-05435a68e7d3)

![Screenshot 2023-09-25 175122](https://github.com/Roselinjovita/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119104296/4514c07c-d7f7-41d6-b6b5-7a7b632edcea)




![Screenshot 2023-09-25 175154](https://github.com/Roselinjovita/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119104296/a8af9b5d-e5ef-4d64-9cff-78fd37b38ce6)




![Screenshot 2023-09-25 175218](https://github.com/Roselinjovita/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119104296/2e5fe8ee-4633-47b8-83fa-e0583fb12d20)







## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

