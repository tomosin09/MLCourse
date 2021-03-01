import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

inv = pd.read_csv('invasion.csv')
x_test = pd.read_csv('operative_information.csv')
X = inv.drop('class', axis=1)
y = inv['class']
parameters = {'n_estimators': range(10, 61, 10),
              'max_depth': range(1, 13, 2)}
rf = RandomForestClassifier(random_state=0)
grid_search_cv_clf = GridSearchCV(rf, parameters, cv=5, n_jobs=-1)
grid_search_cv_clf.fit(X, y)
best_clf = grid_search_cv_clf.best_estimator_
best_clf.fit(X, y)
y_pred = best_clf.predict(x_test)
values, counts = np.unique(y_pred, return_counts=True)
feature_importances = best_clf.feature_importances_
feature_importances_df = pd.DataFrame({'features': list(X),
                                       'feature_importances': feature_importances})
print(feature_importances_df.sort_values('feature_importances', ascending=False))

