from sklearn import tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

df = pd.read_csv('dogs_n_cats.csv')
df2 = pd.read_json('dataset_209691_15.txt')
y = df['Вид']
y = pd.get_dummies(y)
y = y.drop('собачка', axis=1).rename(columns={'котик': 'animal'})
X = df.drop('Вид', axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
scores_data = pd.DataFrame()
max_depth_values = range(1, 100)
for max_depth in max_depth_values:
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    mean_cross_val_score = cross_val_score(clf, X_train, y_train, cv=5).mean()
    temp_scores_data = pd.DataFrame({'max_depth': [max_depth]
                                        , 'train_score': [train_score]
                                        , 'test_score': [test_score]
                                        , 'cross_val_score': [mean_cross_val_score]})
    scores_data = scores_data.append(temp_scores_data)
# print(scores_data.columns)
best_depth = scores_data.groupby(
    'max_depth', as_index=False).agg(
    {'cross_val_score' : 'max'}).max_depth.iloc[0]
best_clf= tree.DecisionTreeClassifier(criterion='entropy', max_depth=best_depth)
clf.fit(X_train, y_train)
# tree.plot_tree(clf.fit(X_train, y_train), filled = True,
#                class_names = ['Кошка','Собака'],
#                feature_names=list(X))
l = list(clf.predict((df2)))
l1 = l.count(1)
print(l1)
# plt.show()


