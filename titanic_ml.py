from sklearn import tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier

titanic_data = pd.read_csv('titanic_dataset/train.csv')
X = titanic_data.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin']
                      , axis=1)
X = pd.get_dummies(X)
X = X.fillna({'Age': X.Age.median()})
y = titanic_data.Survived
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

scores_data_long = pd.melt(scores_data, id_vars=['max_depth'],
                           value_vars=['train_score', 'test_score', 'cross_val_score'],
                           var_name='set_type', value_name='score')
clf_rf = RandomForestClassifier()
parameters = {'n_estimators': [10, 20, 30], 'max_depth': [2, 5, 7, 10]}
# clf = tree.DecisionTreeClassifier()
# parameters = {'criterion': ['gini', 'entropy']
#     , 'max_depth': range(1, 30)}
# grid_search_cv_clf = GridSearchCV(clf, parameters, cv=5)
grid_search_cv_clf = GridSearchCV(clf_rf, parameters, cv=5)
grid_search_cv_clf.fit(X_train, y_train)
# print(grid_search_cv_clf.best_params_)
best_clf = grid_search_cv_clf.best_estimator_
print(best_clf.score(X_test, y_test))
feature_importances = best_clf.feature_importances_
feature_importances_df = pd.DataFrame({'features': list(X_train),
                                       'feature_importances': feature_importances})
print(feature_importances_df.sort_values('feature_importances', ascending=False))
# y_pred = best_clf.predict(X_test)
# recall_score(y_test, y_pred)
# y_predicted_prob = best_clf.predict_proba(X_test)
# y_pred_2 = np.where(y_predicted_prob[:, 1]>0.1, 1, 0)
# print(precision_score(y_test, y_pred_2))
# print(recall_score(y_test, y_pred_2))
#
# fpr, tpr, thresholds = roc_curve(y_test, y_predicted_prob[:,1])
# roc_auc= auc(fpr, tpr)
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange',
#          label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()
