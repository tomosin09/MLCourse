import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
from sklearn.model_selection import cross_val_score

scores_data = pd.DataFrame()
max_depth_values = range(1, 100)
for max_depth in max_depth_values:
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
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
clf = DecisionTreeClassifier(criterion='entropy', max_depth=best_depth)
clf.fit(X_train, y_train)
predictions = clf.predict((X_test))
precision = precision_score(y_test, predictions, average='micro')