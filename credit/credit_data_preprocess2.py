import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from statsmodels.distributions.empirical_distribution import ECDF


def read_data():
    np.set_printoptions(edgeitems=10)
    np.core.arrayprint._line_width = 180
    # 显示为……的列
    pd.set_option('display.max_columns', 20)

    train_data = pd.read_csv("cs-training.csv")
    # test_data=pd.read_csv("cs-test.csv")
    # 判断是否有重复
    train_data.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)
    # test_data.rename(columns={'Unnamed: 0':'ID'},inplace=True)

    # print(train_data.duplicated().value_counts())
    # print(test_data.duplicated().value_counts())
    # print(train_data.describe())
    # 收入的数量120269不对 家属人数146076也不对

    df_mis_inc = train_data[train_data['MonthlyIncome'].isna()]  # 判断对应的矩阵为空则赋true，size相同
    df_not_mis_inc = train_data[train_data['MonthlyIncome'].notna()]  # 判断收入没丢的
    varNames = ['RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio',
                'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
                'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents']

    return df_mis_inc, df_not_mis_inc, varNames

def visualizeECDF(variable, data):
    df = data[:]  # 入参是一个向量。故[:]： for a (say) NumPy array, it will create a new view to the same data.
    ecdf = ECDF(df[variable])
    x = np.linspace(min(df[variable]), np.percentile(df[variable], 99.9))
    y = ecdf(x)
    plt.step(x, y)

def debitRatio():
    df = pd.read_csv("cs-training.csv")
    perc = range(81)
    perc = [10, 20, 30, 40, 50, 60, 70, 80]
    val = []
    for i in perc:
        val.append(np.percentile(df['DebtRatio'], i))
    plt.plot(perc, val, 'go-', linewidth=2, markersize=12)

def debtRatioAboutIncome():
    df = pd.read_csv("cs-training.csv")
    df_not_mis_inc = df[df['MonthlyIncome'].notna()]
    df_mis_inc = df[df['MonthlyIncome'].isna()]
    perc1 = [99.0, 99.1, 99.2, 99.3, 99.4, 99.5, 99.6, 99.7, 99.8, 99.9]
    perc2 = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    val1 = []
    val2 = []
    for i in perc1:
        val1.append(np.percentile(df_not_mis_inc['DebtRatio'], i))
    for i in perc2:
        val2.append(np.percentile(df_mis_inc['DebtRatio'], i))
    plt.plot(perc1, val1)
    plt.show()
    plt.plot(perc2, val2)
    plt.show()

def showVariable():
    # 类中函数调用其他函数
    df_mis_inc, df_not_mis_inc, var_names = read_data()
    print(type(df_mis_inc))
    print(df_not_mis_inc)

def test():
    df = pd.DataFrame(
        {'age': [5, 6, np.NaN],
         'born': [pd.NaT, pd.Timestamp('1939-05-27'), pd.Timestamp('1940-04-25')],
         'name': ['Alfred', 'Batman', ''],
         'toy': [None, 'Batmobile', 'Joker']})
    print(df)
    print(df['name'].isna)
    print(df[df['name'].isna])



