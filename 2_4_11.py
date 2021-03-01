from sklearn import tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

df_train = pd.read_csv('train_iris.csv')
df_test = pd.read_csv('test_iris.csv')
X_train = df_train.drop(['species', 'Unnamed: 0']
                      , axis=1)
y_train = df_train.species
X_test = df_test.drop(['species', 'Unnamed: 0']
                      , axis=1)
y_test = df_test.species
rs = np.random.seed(0)
scores_data = pd.DataFrame()
max_depth_values = range(1, 100)
for max_depth in max_depth_values:
    my_awesome_tree = tree.DecisionTreeClassifier(
        random_state=rs, criterion='entropy',max_depth=max_depth)
    my_awesome_tree.fit(X_train, y_train)
    train_score = my_awesome_tree.score(X_train, y_train)
    test_score = my_awesome_tree.score(X_test, y_test)
    # mean_cross_val_score = cross_val_score(my_awesome_tree, X_train, y_train, cv=5).mean()
    temp_scores_data = pd.DataFrame({'max_depth': [max_depth]
                                        , 'train_score': [train_score]
                                        , 'test_score': [test_score]})
    scores_data = scores_data.append(temp_scores_data)
scores_data_long = pd.melt(scores_data, id_vars=['max_depth'],
                           value_vars=['train_score', 'test_score'],
                           var_name='set_type', value_name='score')
sns.lineplot(x='max_depth', y='score', hue='set_type', data=scores_data_long)
plt.show()
