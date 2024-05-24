# Stochastic Gradient Descent Classifier/Regressor Examples

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import datasets
import numpy as np

#load iris database
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

# training and testing errors
def sgd_classifier(X,y):
    X_train,X_test,y_train,y_test = train_test_split(X,y)
    clf = SGDClassifier(loss='hinge', penalty='l2', max_iter=5)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(y_test)
    r2 = metrics.r2_score(y_test,y_pred)
    return y_pred, r2, clf.coef_, clf.intercept_

#Plotting
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay

#shuffle 
idx = np.arange(X.shape[0])
np.random.seed(13)
np.random.shuffle(idx)
X = X[idx]
y = y[idx]
colors = "b

# standardize
mean = X.mean(axis =0)
std = X.std(axis =0)
X = (X - mean)/ std

clf = SGDClassifier(alpha=0.01, max_iter=100).fit(X,y)
ax = plt.gca()
DecisionBoundaryDisplay.from_estimator(
    clf,
    X,
    cmap=plt.cm.Paired,
    ax=ax,
    response_method="predict",
    xlabel=iris.feature_names[0],
    ylabel=iris.feature_names[1],
)
plt.axis("tight")

# Plot also the training points
for i, color in zip(clf.classes_, colors):
    idx = np.where(y == i)
    plt.scatter(
        X[idx, 0],
        X[idx, 1],
        c=color,
        label=iris.target_names[i],
        cmap=plt.cm.Paired,
        edgecolor="black",
        s=20,
    )
plt.title("Decision surface of multi-class SGD")
plt.axis("tight")


# Plot the three one-against-all classifiers
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
coef = clf.coef_
intercept = clf.intercept_


def plot_hyperplane(c, color):
    def line(x0):
        return (-(x0 * coef[c, 0]) - intercept[c]) / coef[c, 1]

    plt.plot([xmin, xmax], [line(xmin), line(xmax)], ls="--", color=color)


for i, color in zip(clf.classes_, colors):
    plot_hyperplane(i, color)
plt.legend()
plt.show()