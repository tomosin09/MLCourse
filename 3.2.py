import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

heart_df = pd.read_csv('heart.csv')
print(heart_df.head())
X = heart_df.drop('target', axis=1)
y = heart_df.target
# x_train, x_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.33, random_state=42)
np.random.seed(0)
rf = RandomForestClassifier(10, max_depth=5)
rf.fit(X,y)
imp = pd.DataFrame(rf.feature_importances_, index=X.columns, columns=['importance'])
imp.sort_values('importance').plot(kind='barh', figsize=(12, 8))
plt.show()