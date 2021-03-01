import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

mush = pd.read_csv('training_mush.csv')
x_test = pd.read_csv('testing_mush.csv')
y_test = pd.read_csv('testing_y_mush.csv')
X = mush.drop('class', axis=1)
y = mush['class']
parameters = {'n_estimators': range(10, 51, 10),
              'max_depth': range(1, 13, 2), 'min_samples_leaf': range(1, 8),
              'min_samples_split': range(2, 10, 2)}
rf = RandomForestClassifier(random_state=0)
grid_search_cv_clf = GridSearchCV(rf, parameters, cv=3, n_jobs=-1)
grid_search_cv_clf.fit(X, y)
print(grid_search_cv_clf.best_params_)
best_clf = grid_search_cv_clf.best_estimator_
best_clf.fit(X, y)
feature_importances = best_clf.feature_importances_
feature_importances_df = pd.DataFrame({'features': list(X),
                                       'feature_importances': feature_importances})
# print(feature_importances_df.sort_values('feature_importances', ascending=False))
y_pred = best_clf.predict(x_test)
count_pred = np.count_nonzero(y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

