import pandas as pd
import matplotlib.pyplot as np

data = pd.read_csv('output/WarAndPeaceCharCtn.csv')

print(data.head())
print(data.tail())

print(data.describe())
data.hist(figsize=(25, 20))
np.show()
