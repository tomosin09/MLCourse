import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


df = pd.read_csv('space_can_be_a_dangerous_place.csv')
X = df.drop('dangerous', axis = 1)
y = df.dangerous
parameters = {'n_estimators': range(10, 61, 10),
              'max_depth': range(1, 13, 2)}
rf = RandomForestClassifier(random_state=0)
grid_search_cv_clf = GridSearchCV(rf, parameters, cv=5, n_jobs=-1)
grid_search_cv_clf.fit(X, y)
best_clf = grid_search_cv_clf.best_estimator_
best_clf.fit(X, y)
feature_importances = best_clf.feature_importances_
feature_importances_df = pd.DataFrame({'features': list(X),
                                       'feature_importances': feature_importances})
print(feature_importances_df.sort_values('feature_importances', ascending=False))