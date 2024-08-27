# Machine Learning Models with Examples

This repository contains a set of Jupyter notebooks, as well as corresponding Python files, each of which explaining a well-known Machine Learning model. We mostly use built-in models from standard libraries such as PyTorch, Scikit-learn, Pandas, Numpy, etc. Each notebook cites references to the theoretical background of the models.

For our examples, we consider either online datasets included in some libraries (e.g., Iris) or we provide the reader with a link to the open-source dataset used.

## Table of Contents

- [Decision Trees](#decision-trees)
- [Train/Test Split](#train-test-split)
- [Linear Regression](#linear-regression)
- [K-Nearest Neighbors (KNN)](#knn)
- [Kernel Methods](#kernel-methods)
- [Cross-Validation](#cross-validation)
- [Gradient Descent](#gradient-descent)
- [Stochastic Gradient Descent (SGD)](#sgd)
- [Support Vector Machines (SVM)](#svm)
- [AdaBoost](#adaboost)
- [Underfitting and Overfitting](#underfitting-and-overfitting)
- [Bagging](#bagging)
- [Neural Networks](#neural-networks)
- [Convolutional Neural Networks (CNN)](#cnn)

## Decision Trees

**Directory: `Decision Trees`**

- Experiments with Decision Tree Classifier from the sklearn library, optimization, visualization, etc.

## Linear Regression

**Files: `Linear_regression.ipynb` and `Linear_regression.py`**

- Basic implementation from scratch of Linear Regression and R² score with tests.
- Using Linear Regression built-in method in the scipy library.
- Using Linear Regression method in the sklearn library with examples of their performance.

## K-Nearest Neighbors (KNN)

**File: `knn.ipynb`**

- Examples of using the built-in K-Nearest Neighbors (KNN) model in the sklearn library with visualization of results.

## Kernel Methods

**Files: `kernel.ipynb` and `kernel.py`**

- Experiments with kernel methods on Ridge Linear Regression/Classifier.

## Cross-Validation

**File: `cross-val.ipynb`**

- Experiments with the Iris dataset using the Decision Tree Classifier in the sklearn library.
- Performing cross-validation.

## Gradient Descent

**Directory: `Gradient Descent`**

- **Gradient_Descent.ipynb** and **Gradient_Descent.py** include:
  - Gradient Descent Algorithm implementation from scratch.

- **Gradient_Descent_Examples.ipynb** includes:
  - Gradient Descent Algorithm implementation using the backward method in PyTorch (built-in method for backpropagation).
  - Using built-in SGD in PyTorch.

- **backpropagation.py** includes:
  - Implementation of the backward method from scratch.

## Stochastic Gradient Descent (SGD)

**Files: `sgd.ipynb` and `sgd.py`**

- Using the built-in Stochastic Gradient Descent Classifier in the sklearn library.
- Experiments with the Iris dataset with visualization of the results.

## AdaBoost

**Files: `adaboost.ipynb` and `adaboost.py`**

- Experiments with the built-in AdaBoost Classifier from the sklearn Library over the Iris dataset.

## Bagging

**Files: `Bagging.ipynb` and `Bagging.py`**

- Experiments with the Bagging (Bootstrap Aggregation) Classifier built into the sklearn library.
- Comparison with the Decision Tree classifier.
- Visualizing Decision Tree results and showcasing Out-of-Bag Observations.

## Neural Networks

**File: `neural-net.ipynb`**

- Basic examples of building Neural Networks using Keras in the TensorFlow library.
- Experiments on a car price dataset.
- Testing validation of the designed models.
- Checking for underfitting and overfitting.

## Convolutional Neural Networks (CNN)

**File: `CNN.ipynb`**

- Designing a Convolutional Neural Network (CNN) using Keras in the TensorFlow library.
- Experiments on the CIFAR-10 Dataset, visualizing the performance of the model.

## Support Vector Machines (SVM)

**Files: `svm.ipynb` and `svm.py`**

- Using several built-in Support Vector Machine Classifiers and regressors (SVC, NuSVC, and LinearSVC) on a simple dataset.

## Train/Test Split

**Files: `Train_Test.ipynb` and `Train_Test.py`**

- Basics of splitting the data into training and testing sets manually or via built-in methods in sklearn.
- Experiments with random examples and visualization of results on a polynomial regressor.

## Underfitting and Overfitting

**File: `under-overfit.ipynb`**

- Building a function that recognizes underfitting and overfitting using mean absolute error on a built-in Decision Tree Regressor in the sklearn library.
- Experiments using the Iris dataset.
