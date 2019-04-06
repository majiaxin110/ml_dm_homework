import numpy as np
import pandas as pd

size = 300
x = np.random.normal(0, 1, size)
df = pd.DataFrame({"A": x, "B": 2 * x + np.random.normal(0, 1, size),
                   "C": x ** 2 + np.random.normal(0, 2, size)})
# print(df)
print("-------------- Pearson --------------")
print(df.corr())
print("------------ Kendall Tau ------------")
print(df.corr("kendall"))
print("------------ Spearman ------------")
print(df.corr("spearman"))
