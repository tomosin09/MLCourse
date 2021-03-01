import pandas as pd
import numpy as np

d = {'type': ['A', 'A', 'B', 'B'], 'value': [10, 14, 12, 23]}
my_data = pd.DataFrame({'type': ['A', 'A', 'B', 'B'], 'value': [10, 14, 12, 23]})

my_stat = pd.read_csv('my_stat.csv')
subset_1 = my_stat.iloc[0:10, [0, 2]].head(10)
subset_2 = my_stat.drop([1, 5], axis=0).iloc[:, [1, 3]]
subset_1 = my_stat.query("V1 > 0 & V3 == 'A'")
subset_2 = my_stat.query("V2 != 10 | V4 >= 10")
my_stat['V5'] = my_stat.V1 + my_stat.V4
my_stat = my_stat.assign(V6=np.log(my_stat['V2']))

my_stat1 = pd.read_csv('my_stat_1.csv')
my_stat1['session_value'] = my_stat1['session_value'].fillna(0)
Mn = my_stat1[my_stat1.n_users >= 0.0].n_users.median()
my_stat1.loc[my_stat1['n_users']<0.0, 'n_users'] = Mn
print(my_stat1.groupby('group', as_index=False).agg({'session_value':'mean'}).rename(columns={'session_value':'mean_session_value'}))

