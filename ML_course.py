from sklearn import tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.DataFrame({'X_1': [1, 1, 1, 0, 0, 0, 0, 1],
                     'X_2': [0, 0, 0, 1, 0, 0, 0, 1],
                     'Y': [1, 1, 1, 1, 0, 0, 0, 0]})
clf = tree.DecisionTreeClassifier(criterion='entropy')
X=data[['X_1', 'X_2']]
y=data.Y
clf.fit(X,y)
tree.plot_tree(clf.fit(X,y), filled = True,
               class_names = ['negative','positive'],
               feature_names=list(X))
plt.show()
