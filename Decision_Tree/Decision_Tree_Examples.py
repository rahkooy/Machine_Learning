# Decision Trees Construction

"""-Task: Given a data set with features (Age, Experience, Rank, nationality), decide if a person should go to a show or not, i.e. Target (Go)

-Source:
https://www.w3schools.com/python/python_ml_decision_tree.asp
"""

# Import pandas and read the data file
import pandas as pd
df = pd.read_csv("data.csv")
df

""" In a decision tree, all values must be numerical, 
 so we change last two non-numerical columns.
 For this we define one dictionary per column: """
d = {'UK':0, 'USA':1, 'N':2}
"map method replaces a dictionary with a column:"
df['Nationality'] = df['Nationality'].map(d)

print(df)

"""Similarly for the last column:"""
d = {'YES': 1, 'NO': 0}
df['Go'] = df['Go'].map(d)

df

# Buil Feature and Targets
features = ['Age', 'Experience', 'Rank', 'Nationality']
X = df[features]
Y = df['Go']

print(X)
print(Y)

# Create Decision Tree via sklearn library:
from sklearn import tree
dtree = tree.DecisionTreeClassifier()
dtree = dtree.fit(X, Y)

# plot decision tree
import matplotlib.pyplot as plt
tree.plot_tree(dtree, feature_names=features)

## Result Explained:
"""Rank <= 6.5 means that every comedian with a rank of 6.5 or lower will follow the True arrow (to the left), and the rest will follow the False arrow (to the right).

Gini = 1 - (x/n)^2 - (y/n)^2, where x is the number of positive answers("GO"), n is the number of samples, and y is the number of negative answers ("NO").


samples = number of comedians left at this point in the decision.

value = [6, 7] means that of these 13 comedians, 6 will get a "NO", and 7 will get a "GO".
"""

## Predicting Values

# Should someone w Age=40, experience=10, rank=7 go to the show?
print(dtree.predict([[40, 10, 7, 1]]))

print(dtree.predict([[40, 10, 6, 1]]))

