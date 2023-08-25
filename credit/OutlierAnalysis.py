import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from statsmodels.distributions.empirical_distribution import ECDF
def outlierAnalysis():
    df=pd.read_csv("cs-training.csv")
    df1=pd.read_csv("cs-training.csv")
    cols = df1.columns
    for i in cols:
        # %%
        df = pd.read_csv("cs-training.csv")
        df_25=np.percentile(df['age'],25)
        df_75=np.percentile(df['age'],75)
        sub=np.subtract(df_25,df_75)
        print(25-1.5*np.subtract(*np.percentile(df['age'], [75, 25])))
        # %%
        a=1e-3
        print(a)
        # %%
        Low_Bound=((np.percentile(df[i],25))- 1.5*np.subtract(*np.percentile(df[i],[75,25])))
        Upp_Bound = ((np.percentile(df[i], 75)) + 1.5 * np.subtract(*np.percentile(df[i], [75, 25])))
        df1[i] = df[i].apply(lambda x: 1 if (x < Low_Bound or x > Upp_Bound) else 0) # 0就是正常值范围内的
        print(df1[i])
        print(df1.sum(axis=0, skipna=True))
if __name__ == '__main__':
    outlierAnalysis()