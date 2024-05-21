# An AdaBoost exammple

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X,y)

def adaboost(X_train,X_test,y_train,y_test,estimater_range): 
    scores =[]
    for n_estimators in estimator_range:
        #create classifier
        clf = AdaBoostClassifier(n_estimators=n_estimators, algorithm="SAMME",)
        #fit data and predict
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        #errors
        scores = cross_val_score(clf, X, y, cv=5)
        scores.mean()
        acc = metrics.accuracy_score(y_test,y_pred)
    return y_pred, scores, acc

estimator_range = [2,4]
adaboost(X_train,X_test,y_train,y_test,estimator_range)
