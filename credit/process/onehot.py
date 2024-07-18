#%%
import pandas as pd
df=pd.read_csv("credit/data/cs-test.csv")
df1=pd.read_csv("credit/data/cs-training-woe.csv")
print(df.dtypes)
print(df1.dtypes)