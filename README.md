Examples in Machine Learning Models

This repository contains a set of Jupyter notesbooks, as well as corresponding python files, each of which explaining a well-known Machine Learning model. We mostly use built-in models in standard libraries, such as PyTorch, Scikit, Pandas, Numpy, etc. The notebooks cite references that provide with theoretical background of the models.

For our examples we consider either online dataset included in some libraries, e.g., Iris, or provide the link to the source of the dataset used.

The content of the files:

-Gradient Descent directory:
   -Gradient_Descent.ipynb and Gradient_Descent.py includes:
      -Gradient Descent Algorithm implementation from scratch
   -Gradient_Descent_Examples.ipynb includes:
      -Gradient Descent Algorithm implementation using backward method in PyTorch (built-in method for backpropogation)
      -Using built-in SGD in PyTorch
   -backpropogation.py includes:
      -implementation of backward method from scratch
   
-Decision Trees Directory includes:
   -experiments with Decision Tree Classifier from sklearn library, optimisatio, visualisation, etc.

-daboost.ipynb and adaboost.py includes:
   -Experiments with built-inAdaBoost Classifier from the sklearn Library over iris dataset 

-Bagging.ipynb and Bagging.py includes:
   -Experiments with Bagging (Bootstrap Aggregation) Classfier built in within sklearn library, comparison with decision tree classifier, visualising Decision tree results and showcasing Out-of_Bag Observations
   
-CNN.ipynb includes:
   -Designing a Convolutional Neural Network (CNN) using Keras in Tensorflow library, and Experiments on CIFAR-10 Dataset visualising the performance of the model

-cross-val.ipynb includes:
   -Experiments with iris dataset using Decision Tree Classifier in sklearn library and doing cross validation.

-kernel.ipynb and kernel.py includes:
   -Experiments with kernel method on Ridge Linear Regression/Classifier

-knn.ipynb includes:
   -Examples of using the built in K-Nearest Neighbourhood (KNN) model in sklearn library with visualisation of results

-Linear_regression.ipynb and Linear_regression.py includes:
   -A basic implementation from scratch of Linear Regression and R2 squared with tests
   -Using Linear Regression built-in method in scipy library
   -Using Linear Regression method in sklearn library with examples on their performance

-neural-net.ipynb includes:
   -Basic Examples on building Neural Networks using Keras in Tensorflow library
   -Experiments on a car price repository
   -Testing validation of the models designed
   -Checking Under/Overfitting

-sgd.ipynb and sgd.py includes:
   -Using built-in Stochastic Gradient Descent Classifier in sklearn linrary, experiments with iris dataset with visualisation of the results

-svm.ipynb and svm.py includes:
   -Using several built-in Support Vector Machine Classifiers and regressors (SVC, NuSVC and LinearSVC) on a simple data

-Train_Test.ipynb and Train_Test.py includes:
   -Basics of splitting the data into training and testing sets manually or via bult-in methods in sklearn, experiments with random examples and visualisation of results on a polynomial regressor

-under-overfit.ipynb includes:
   -bulding a function that recognises underfitting and overfitting using mean absolute error on a built-in Decision Tree Regressor in sklearn library
   -experimens using iris dataset

-
