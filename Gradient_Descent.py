# Gradient Descent Basic Implementation

# source: https://github.com/patrickloeber/pytorchTutorial
# Compare w
# 1-backpropogation.py for an implementation using Backward method
# 2-Linear_regression.py for an object-orinted implementation

import numpy as np 

# Linear regression with bias=0
# f = w * x 

# Lwt w=2, so f = 2 * x
X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)

#Set w = 0
w = 0.0

# model output
def forward(x):
    return w * x

# loss function = MSE (mean squared error) = 
def loss(y, y_pred):
    return ((y_pred - y)**2).mean()

# J = MSE = 1/N * (w*x - y)**2
# dJ/dw = 1/N * 2x(w*x - y)
def gradient(x, y, y_pred):
    return np.mean(2*x*(y_pred - y))

print(f'Prediction before training: f(5) = {forward(5):.3f}')

# Training
learning_rate = 0.01
n_iters = 20

for epoch in range(n_iters):
    # predict = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)
    
    # calculate gradients
    dw = gradient(X, Y, y_pred)

    # update weights
    w -= learning_rate * dw

    if epoch % 2 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')
     
print(f'Prediction after training: f(5) = {forward(5):.3f}')