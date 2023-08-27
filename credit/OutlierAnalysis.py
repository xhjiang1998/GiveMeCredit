import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from statsmodels.distributions.empirical_distribution import ECDF
# 关于本函数的数理逻辑链接：https://zhuanlan.zhihu.com/p/34073898
# 计算分位数的方法：https://zhuanlan.zhihu.com/p/235345817

#加载数据
def load_data():
    df=pd.read_csv("cs-training.csv")
    df1=pd.read_csv("cs-training.csv")
    cols = df1.columns
    return df,df1,cols

# 利用上下分位数计算正常值范围，统计异常特征值的数量
def outlierAnalysis():
    df=pd.read_csv("cs-training.csv")
    df1=pd.read_csv("cs-training.csv")
    cols = df1.columns
    np.set_printoptions(suppress=True)
    pd.set_option('display.float_format', lambda x: '%.2f' % x)

    for i in cols:
        # # %%
        # df = pd.read_csv("cs-training.csv")
        # df_25=np.percentile(df['age'],25)
        # df_75=np.percentile(df['age'],75)
        # sub=np.subtract(df_25,df_75)
        # print(25-1.5*np.subtract(*np.percentile(df['age'], [75, 25])))
        # # %%
        # a=1e+3
        # print(a)
        # # %%
        Low_Bound=((np.percentile(df[i],25))- 1.5*np.subtract(*np.percentile(df[i],[75,25])))
        Upp_Bound = ((np.percentile(df[i], 75)) + 1.5 * np.subtract(*np.percentile(df[i], [75, 25])))
        df1[i] = df[i].apply(lambda x: 1 if (x < Low_Bound or x > Upp_Bound) else 0) # 0就是正常值范围内的
        df2 = df1.sum(axis=0, skipna=True)
        print(df2) #行累加，按照列归纳出结果，跳过为null值的
    print(df2.sort_values());

# 计算部分特征对应的特征值
def value_count():
    df=load_data()
    print(pd.value_counts(df['NumberOfTime30-59DaysPastDueNotWorse']))

# 描绘相关性，观察
def outilerprocess(variable,data):
    varNames = ['RevolvingUtilizationOfUnsecuredLines', 'DebtRatio', 'MonthlyIncome']
    df = data[:]
    ecdf = ECDF(df[variable])
    x = np.linspace(min(df[variable]), np.percentile(df[variable],99.5))
    y = ecdf(x)
    plt.step(x, y)
    np.random.seed(100)
    fig, axes = plt.subplots(nrows=3, ncols=2)
    fig.tight_layout()
    fig.set_figheight(10)
    fig.set_figwidth(15)
    plt.subplots_adjust(hspace=0.5)
    ax = plt.subplot(3,2,1)
    ax.set_title(varNames[0])
    outilerprocess(varNames[0], data = df)
    perc = [97.5,98,98.5,99,99.5,100]
    val = []
    for i in perc:
    val.append(np.percentile(df['RevolvingUtilizationOfUnsecuredLines'],i ))
    ax = plt.subplot(3,2,2)
    ax.set_title(varNames[0])
    plt.plot(perc, val, 'go - ', linewidth=2, markersize=12)
    ax = plt.subplot(3,2,3)
    ax.set_title(varNames[1])
    outilerprocess(varNames[1], data = df)
    perc = [97.5,98,98.5,99,99.5,100]
    val = []
    for i in perc:
        val.append(np.percentile(df['DebtRatio'],i ))
        ax = plt.subplot(3,2,4)
        ax.set_title(varNames[1])
        plt.plot(perc, val, 'go - ', linewidth=2, markersize=12)
    
    ax = plt.subplot(3,2,5)
    ax.set_title(varNames[2])
    outilerprocess(varNames[2], data = df)
    perc = [97.5,98,98.5,99,99.5,100]
    val = []
    for i in perc:
        val.append(np.percentile(df['MonthlyIncome'],i ))
        ax = plt.subplot(3,2,6)
        ax.set_title(varNames[2])
        plt.plot(perc, val, 'go - ', linewidth=2, markersize=12)

if __name__ == '__main__':
    value_count()