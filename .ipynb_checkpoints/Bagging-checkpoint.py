# A Bootstrao Aggregation (Bagging) Function vs Decision Tree

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Sklearn's dataset: some small datasets
df = datasets.load_wine(as_frame=True) 
# as_frame is for not loosing the feature name

#print(df)
X = df.data
print(X.columns)
y = df.target
print(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size =0.25, random_state=22 )

#Decision Tree Classifier
dtree = DecisionTreeClassifier(random_state=22)
dtree.fit(X_train,y_train)

#predicting and checking accuracy
y_pred_train = dtree.predict(X_train)
train_accu = accuracy_score(y_train,y_pred_train)
print('Trained data accuracy= ',train_accu)
y_pred_test = dtree.predict(X_test)
test_acc = accuracy_score(y_test,y_pred_test)
print('Test Data accuracy= ',test_acc)

#====================================================
## Bagging
from sklearn.ensemble import BaggingClassifier

# values for number of estimators
estimator_range = [2,4,6,8,10,12,14,16]

models = []
scores = []

for n_estimators in estimator_range:
    
    # Create bagging classifier
    clf = BaggingClassifier(n_estimators=n_estimators, 
                            random_state=22)
    # Fit the model & predict
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    
    # Accuracy
    acc = accuracy_score(y_test,y_pred)
    print('For estimator=', n_estimators, 
          'accuracy score is: ', acc)
    
    # Create list of model and accuracy scores
    models.append(clf)
    
    scores.append(acc)

print(Output Explained: By change of estimator the accuracy changes)

## Visualising Improvemnt in Accuracy Score
import matplotlib.pyplot as plt

plt.figure(figsize=(9,6))
plt.plot(estimator_range,scores)

plt.xlabel('n_estimator')
plt.ylabel('scores')

plt.show

### Out-of-Bag Observations (obb)
""" As bootstrapping chooses random subsets of observations to create 
classifiers, there are observations that are left out in the selection 
process.obb_score=True evalues the model with out-of-bag score.
"""

oob_model = BaggingClassifier(n_estimators=12, oob_score=True,
                             random_state=22)
# n_estimators=12 was the best in the previous run
oob_model.fit(X_train,y_train)
score= oob_model.oob_score_
print(score)

## Visualising Decision Tress in the Bagging Classifier
from sklearn.tree import plot_tree

plt.figure(figsize=(30,20))
plot_tree(clf.estimators_[0], feature_names = X.columns)

