import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from statsmodels.distributions.empirical_distribution import ECDF
from OutlierAnalysis import load_data

def read_data():
    np.set_printoptions(edgeitems=10)
    np.core.arrayprint._line_width = 180
    # 显示为……的列
    pd.set_option('display.max_columns', 20)

    train_data = pd.read_csv("cs-training.csv")
    # 判断是否有重复
    train_data.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)

    # print(train_data.duplicated().value_counts())
    # print(test_data.duplicated().value_counts())
    # print(train_data.describe())
    # 收入的数量120269不对 家属人数146076也不对


# 计算部分特征对应的特征值
def adjust_value_count():
    df,df1,cols=load_data()
    print(df['NumberOfTime30-59DaysPastDueNotWorse'].value_counts())
    print(df['NumberOfTimes90DaysLate'].value_counts())
    print(df['NumberOfTime60-89DaysPastDueNotWorse'].value_counts())

    # 将部分明显的偏差正常的值纠正
    df.loc[df['NumberOfTimes90DaysLate']>20,'NumberOfTimes90DaysLate']=20
    df.loc[df['NumberOfTime60-89DaysPastDueNotWorse']>20,'NumberOfTime60-89DaysPastDueNotWorse']=20
    df.loc[df['NumberOfTime30-59DaysPastDueNotWorse']>20,'NumberOfTime30-59DaysPastDueNotWorse']=20
    # 观察结果(过去两年中发生60-89天逾期的次数)
    print(df['NumberOfTime60-89DaysPastDueNotWorse'].value_counts())

    # 缺失值填补
    df_mis_inc = df[df['MonthlyIncome'].isna()]  # 只输出为true的df，而true就是MonthlyIncome为空的
    df_mis_inc['MonthlyIncome']=0

if __name__ == '__main__':
     read_data()
     adjust_value_count()



