import pandas as pd
import numpy as np
from time import time

iris = pd.read_csv('testing_mush.csv')
before = time()
iris.apply('mean')
after = time()
print(after - before)
before = time()
iris.apply(np.mean)
after = time()
print(after - before)
before = time()
iris.mean(axis=0)
after = time()
print(after - before)
before = time()
iris.describe().loc['mean']
after = time()
print(after - before)