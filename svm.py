# Support Vector Machines (SVMs)
# source: https://scikit-learn.org/stable/modules/svm.html

## Classifiers: SVC, NuSVC, LinearSVC

from sklearn import svm

def svm_svc(X_train,y_train, y_test):
    clf = svm.SVC()
    clf.fit(X,y)
    y_pred = slf.predict(y_test)
    supp_vec = clf.support_vectors_
    supp = clf.support_
    return y_pred, supp_vec, supp

X_train = [[0,1],[1,1]]
y_test = [2,2]
y_train = [0,1]
print('y_pred is ', svm_svc(X,y))

## Regressors: SVR, NuSVR, LinearSVR

def svm_svr(X_train,y_train,y_test):
    regr = svm.SVR()
    regr.fit(X_train,y_train)
    y_pred = regr.predict(y_test)
    return y_pred

print('y_regressor is ', svm_svr(X_train,y_train,y_test))

