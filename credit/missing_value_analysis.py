import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    x = np.linspace(min(df[variable]), np.nanpercentile(df[variable], 99.9))
    y = ecdf(x)
    plt.step(x, y)

def show():
    df_mis_inc, df_not_mis_inc, varNames=read_data()
    np.random.seed(100)
    fig, axes = plt.subplots(nrows=9, ncols=2)  #9行两列的图
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
    # debitRatio在df_not_mis_inc和df_mis_inc中的经验分布函数体现的差距较大

def debitRatio():
    df = pd.read_csv("cs-training.csv")
    perc = range(81)
    val = []
    for i in perc:
        val.append(np.percentile(df['DebtRatio'], i))
    plt.plot(perc, val, 'go-', linewidth=2, markersize=12)

# 显示负债率最高的有收入的人 的 top1%的负债率曲线
    # 和收入缺失的人的 整体负债率 大致相当
    # 故用前者的收入来替代后者，就归0吧
def debtRatioAboutIncome():
    df_mis_inc, df_not_mis_inc, var_names = read_data()
    perc1 = [99.0, 99.1, 99.2, 99.3, 99.4, 99.5, 99.6, 99.7, 99.8, 99.9]
    perc2 = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    val1 = []
    val2 = []
    for i in perc1:
        val1.append(np.percentile(df_not_mis_inc['DebtRatio'], i))
        #给出来的值是超过i%的值，该值是属于df_not_mis_inc['DebtRatio']的一项值
    for i in perc2:
        val2.append(np.percentile(df_mis_inc['DebtRatio'], i)) # 无收入，负债比例从0到90的人的经验曲线
    plt.plot(perc1, val1)
    plt.plot(perc2, val2)
    plt.show()



if __name__ == '__main__':
    debtRatioAboutIncome()



