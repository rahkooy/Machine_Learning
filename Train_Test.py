"""Train/Test

source: https://www.w3schools.com/python/python_ml_train_test.asp

1-split the data into training and testing sets

2-find a model using training set (e.g., via regression)

3-check if the models is OK for training data (e.g. for regression use R-Squared value for testing data)

4-Check if the model is OK for testing data (e,g, R-squared for testing data)
"""

import numpy as np
import matplotlib.pyplot as plt

"""seed makes the same random number to be the generated every time; 
without seed different numbers will be generated"""
np.random.seed(4) 
#normal distribution, center of dist =3, deviation=1, sample size =100:
x = np.random.normal(3, 1, 100) 
y = np.random.normal(150, 40, 100) / x

"""
x = minutes spent before purchasing, 
y = money spent
"""

plt.scatter(x,y)
plt.show()

# split sample into training set (80%) and testing set (20%)
train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]

plt.scatter(train_x, train_y)
plt.show()

plt.scatter(test_x, test_y)
plt.show()

## Model via polynomial regression
# By intuition, training set looks best match a polynomial regression:
mymodel = np.poly1d(np.polyfit(train_x, train_y, 4))
myline = np.linspace(0, 6, 10)  # 0<x<6, 10: number of pieces of line forming curve

plt.scatter(train_x, train_y)
plt.plot(myline, mymodel(myline))
plt.show()

# testing R-squared for training data
from sklearn.metrics import r2_score
r2 = r2_score(train_y, mymodel(train_x))
print(r2)

## Testing model
# via checking if the R-Squared is OK for testing set 

# finding R-squared for testing data
r2test = r2_score(test_y, mymodel(test_x))
print(r2test)
# 0.81 is OK, so model works fine

## Predicting values using model
# check how much a customer spending 5 min in the shop would purchase
print(mymodel(5))
# 30 looks ok!
print(mymodel(2))
print(mymodel(0.5))
print(mymodel(0.1))