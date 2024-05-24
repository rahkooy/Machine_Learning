# Kernel Method on Ridge Linear Regression

"""
https://scikit-learn.org/stable/modules/kernel_ridge.html

Ridge regression: https://scikit-learn.org/stable/modules/linear_model.html#
"""

## Ridge Regression
"""
It's a linear regression w different coefficient minimizsation than ordinary lears square.

The coefficient is min |Xw -y|^2 +alpha |w|^2
"""

# Ridge regression example:
from sklearn import linear_model
X = [[0,0],[0,0],[1,1]]
y = [0,.1,1]

def ridg_reg(X,y):
    reg = linear_model.Ridge(alpha=.5)
    reg.fit(X, y)
    y_pred = reg.predict([0,1])
    return y_pred, reg.coef_, reg.intercept_

### Note. Ridge Classifier: Can be much faster than Logistirc Regression

## Kernel Ridge Regression
"""
model learned by KernelRidge = Support Vector Regression
Loss functions are different
"""

from sklearn.kernel_ridge import KernelRidge
import numpy as np
n_samples, n_features = 10, 5
rng = np.random.RandomState(0)
y = rng.randn(n_samples)
X = rng.randn(n_samples, n_features)

def ker_ridge(X,y,y_test)
    krr = KernelRidge(alpha = 1.0)
    krr.fit(X,y)
    y_pred = krr.predict(y_test)
    return y_pred
