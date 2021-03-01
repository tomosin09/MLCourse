from sklearn import tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

events_data = pd.read_csv('event_data_train.csv')
submission_data = pd.read_csv('submissions_data_train.csv')
submission_data['date'] = pd.to_datetime(submission_data.timestamp, unit='s')
submission_data['day'] = submission_data.date.dt.date
events_data['date'] = pd.to_datetime(events_data.timestamp, unit='s')
events_data['day'] = events_data.date.dt.date
users_events_data = events_data.pivot_table(index='user_id',
                                            columns='action',
                                            values='step_id',
                                            aggfunc='count',
                                            fill_value=0).reset_index()
users_scores = submission_data.pivot_table(index='user_id',
                                           columns='submission_status',
                                           values='step_id',
                                           aggfunc='count',
                                           fill_value=0).reset_index()
gap_data = events_data[['user_id', 'day', 'timestamp']] \
    .drop_duplicates(subset=['user_id', 'day']).groupby('user_id')['timestamp'] \
    .apply(list).apply(np.diff).values
gap_data = pd.Series(np.concatenate(gap_data, axis=0))
gap_da = gap_data / (24 * 60 * 60)
gap_data[gap_data < 200]
gap_quantile = gap_data.quantile(0.98)
users_data = events_data.groupby('user_id', as_index=False) \
    .agg({'timestamp': 'max'}).rename(columns={'timestamp': 'last_timestamp'})
now = 1526772811
drop_out_threshold = 30 * 24 * 60 * 60
users_data['is_gone_user'] = (now - users_data.last_timestamp) > drop_out_threshold
users_data = users_data.merge(users_scores, on='user_id', how='outer').head()
users_data = users_data.fillna(0)
users_data = users_data.merge(users_events_data, how='outer')
users_days = events_data.groupby('user_id').day.nunique().to_frame().reset_index()
users_data = users_data.merge(users_days, how='outer')
users_data['passed_corse'] = users_data.passed > 170
user_min_time = events_data.groupby('user_id', as_index=False) \
    .agg({'timestamp': 'min'}) \
    .rename({'timestamp': 'min_timestamp'}, axis=1)
users_data = users_data.merge(user_min_time, how='outer')
events_data_train = pd.DataFrame()
# for user_id in users_data.user_id:
#     min_user_time = users_data[users_data.user_id == user_id]\
#         .min_timestamp.item()
#     time_threshold = min_user_time +3 * 24 * 60 * 60
#     user_events_data = events_data[(events_data.user_id == user_id)
#                                    & (events_data.timestamp < time_threshold)]
#     events_data_train= events_data_train.append(user_events_data)
events_data = events_data.merge(user_min_time, how='outer')
events_data_train = events_data[events_data['timestamp'] <= events_data['min_timestamp']
                                + 3 * 24 * 60 * 60]
submission_data = submission_data.merge(user_min_time, how='outer')
submission_data_train = submission_data[
    submission_data['timestamp'] <= submission_data['min_timestamp']
    + 3 * 24 * 60 * 60]

X = submission_data_train.groupby('user_id') \
    .day.nunique().to_frame().reset_index() \
    .rename(columns={'day': 'days'})
steps_tried = submission_data_train.groupby('user_id') \
    .step_id.nunique().to_frame().reset_index() \
    .rename(columns={'step_id': 'steps_tried'})
X = X.merge(steps_tried, on='user_id', how='outer')
X = X.merge(submission_data_train.pivot_table(index='user_id',
                                              columns='submission_status',
                                              values='step_id',
                                              aggfunc='count',
                                              fill_value=0).reset_index(), how='outer')
X['correct_ratio'] = X.correct / (X.correct + X.wrong)
X = X.merge(events_data_train.pivot_table(index='user_id',
                                          columns='action',
                                          values='step_id',
                                          aggfunc='count',
                                          fill_value=0).reset_index()
            [['user_id', 'viewed']], how='outer')
X = X.fillna(0)
X = X.merge(users_data[['user_id', 'passed_corse', 'is_gone_user']], how='outer')
X = X[~((X.is_gone_user == False) & (X.passed_corse == False))]
y = X.passed_corse.map(int)
X = X.drop(['passed_corse', 'is_gone_user'], axis=1)
X = X.set_index(X.user_id)
X = X.drop('user_id', axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
clf = RandomForestClassifier()
parameters = {'n_estimators': [50, 75, 100], 'max_depth': [2,5,7,10]}
# clf = tree.DecisionTreeClassifier()
# parameters = {'criterion': ['gini', 'entropy'], 'max_depth': range(1, 30), 'min_samples_split': (2, 10, 50, 100),
#               'min_samples_leaf': (1, 2, 10, 20, 30)}
grid_search_cv_clf = GridSearchCV(clf, parameters, cv=5)
grid_search_cv_clf.fit(X_train, y_train)
best_clf = grid_search_cv_clf.best_estimator_
# plt.figure(figsize=(10, 10))
# tree.plot_tree(best_clf.fit(X_train, y_train), filled=True,
#                class_names=['Покинул', 'Остался'],
#                feature_names=list(X))
# plt.show()
y_pred = best_clf.predict(X_test)
recall_score(y_test, y_pred)
y_predicted_prob = best_clf.predict_proba(X_test)
y_pred_2 = np.where(y_predicted_prob[:, 1] > 0.2, 1, 0)
conf_matrix = confusion_matrix(y_test, y_pred_2)
precision = precision_score(y_test, y_pred_2)
recall = recall_score(y_test, y_pred_2)
accuracy = accuracy_score(y_test, y_pred_2)
print(f'parameters = {grid_search_cv_clf.best_params_}'
      f'\nconfusion matrix = {conf_matrix}'
      f'\naccuracy = {accuracy}'
      f'\nprecision = {precision}'
      f'\nrecall = {recall}')

fpr, tpr, thresholds = roc_curve(y_test, y_predicted_prob[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange',
         label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


'''Давайте найдем такой стэп, используя данные о сабмитах. 
Для каждого пользователя найдите такой шаг, который он не смог решить, 
и после этого не пытался решать другие шаги. 
Затем найдите id шага,  
который стал финальной точкой практического обучения на курсе для максимального числа пользователей. '''
# id_step_wrong = submission_data.pivot_table(index='step_id',
#                                             columns='submission_status',
#                                             values='user_id',
#                                             aggfunc='count',
#                                             fill_value=0).reset_index()
# id_step_wrong = id_step_wrong.groupby('wrong', as_index=False).max()

# users_data[users_data.passed_corse].day.hist()
# plt.show()
# print(users_data.groupby('passed_corse').count())

# Поиск id Анатолия Карпова
# id_karpov = events_data.groupby('user_id')\
#     .agg({'step_id':'count'})
# print(id_karpov.sort_values(['step_id'], ascending = False).head(20))
