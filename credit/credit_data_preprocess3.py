import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from statsmodels.distributions.empirical_distribution import ECDF
from credit_data_preprocess2 import visualizeECDF



def read_data():
    """

    :rtype: 对应的收入为空的dataframe等
    """
    np.set_printoptions(edgeitems=10)
    np.core.arrayprint._line_width = 180
    df = pd.read_csv("cs-training.csv")
    df_mis_inc = df[df['MonthlyIncome'].isna()]  # 只输出为true的df，而true就是MonthlyIncome为空的
    df_not_mis_inc = df[df['MonthlyIncome'].notna()]  # 输出收入没丢的
    varNames = ['RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio',
                'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
                'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents']
    return df_mis_inc, df_not_mis_inc, varNames

def visualizeECDF( variable, data):
    df = data[:]  # 入参是一个向量。故[:]： for a (say) NumPy array, it will create a new view to the same data.
    ecdf = ECDF(df[variable])
    x = np.linspace(min(df[variable]), np.percentile(df[variable], 99.9))
    y = ecdf(x)
    plt.step(x, y)

def show():
    df_mis_inc, df_not_mis_inc, varNames=read_data()
    np.random.seed(100)
    fig, axes = plt.subplots(nrows=9, ncols=2)
    fig.tight_layout()
    fig.set_figheight(45)
    fig.set_figwidth(15)
    plt.subplots_adjust(hspace=0.8)
    for i in [1, 3, 5, 7, 9, 11, 13, 15, 17]:
        ax = plt.subplot(9, 2, i)
        ax.set_title(varNames[(i - 1) // 2])
        visualizeECDF(varNames[(i - 1) // 2], df_not_mis_inc)
        ax = plt.subplot(9, 2, i + 1)
        ax.set_title(varNames[(i - 1) // 2])
        visualizeECDF(varNames[(i - 1) // 2], df_mis_inc)

def debitRatio():
    df = pd.read_csv("cs-training.csv")
    perc = range(81)
    perc = [10, 20, 30, 40, 50, 60, 70, 80]
    val = []
    for i in perc:
        val.append(np.percentile(df['DebtRatio'], i))
    plt.plot(perc, val, 'go-', linewidth=2, markersize=12)

def debtRatioAboutIncome():
    df_mis_inc, df_not_mis_inc, var_names = read_data()
    perc1 = [99.0, 99.1, 99.2, 99.3, 99.4, 99.5, 99.6, 99.7, 99.8, 99.9]
    perc2 = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    perc3 = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    val1 = []
    val2 = []
    val3 = []
    df_val3 = [0,10,20,30,40,50,60,70,80,90]
    print(df_not_mis_inc.shape,'\n')
    print('df_not_mis_inc',df_not_mis_inc['DebtRatio'].shape,"\n")
    print('df_mis_inc','\n',df_mis_inc['DebtRatio'].shape,"\n")
    print('df_not_mis_inc','\n',df_not_mis_inc['DebtRatio'],"\n")


    print('df_mis_inc',np.percentile(df_mis_inc['DebtRatio'],50),'\n')
    for i in perc1:
        val1.append(np.percentile(df_not_mis_inc['DebtRatio'], i))
        #给出来的值是超过i%的值，该值是属于df_not_mis_inc['DebtRatio']的一项值
    for i in perc2:
        val2.append(np.percentile(df_mis_inc['DebtRatio'], i))
        print(i,val2)
    for i in perc3:
        val3.append(np.percentile(i,df_val3))

    plt.plot(perc1, val1)
    plt.show()
    plt.plot(perc2, val2)
    plt.show()
    plt.plot(perc3, val3)
    plt.show()


def printline():
    print('----------------')
def test():
    df = pd.DataFrame({'age': [5, 6, np.NaN],
                       'born': [pd.NaT, pd.Timestamp('1939-05-27'), pd.Timestamp('1940-04-25')],
                       'name': ['Alfred', 'Batman', ''],
                       'toy': [None, 'Batmobile', 'Joker']})
    print(df)
    printline()
    print(df['born'].isna())
    printline()
    print(df[df['born'].isna()])


if __name__ == '__main__':
    debtRatioAboutIncome()



