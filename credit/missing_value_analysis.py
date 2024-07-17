import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF



def read_data():
    """
    :rtype: 对应的收入为空的dataframe等
    """
    np.set_printoptions(edgeitems=10)
    np.core.arrayprint._line_width = 180
    df = pd.read_csv("data/cs-training.csv")
    df_mis_inc = df[df['MonthlyIncome'].isna()]  # 只输出为true的df，而true就是MonthlyIncome为空的
    df_not_mis_inc = df[df['MonthlyIncome'].notna()]  # 输出收入没丢的
    varNames = ['RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio',
                'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
                'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents']
    return df_mis_inc, df_not_mis_inc, varNames

def visualizeECDF(variable, data):
    """
    绘制经验分布函数（ECDF）图。

    参数:
    variable: 字符串，表示数据集中要用于绘制ECDF的变量名。
    data: 数据集，包含多个变量和对应值的DataFrame。

    返回值:
    无返回值，该函数直接显示绘制的ECDF图。
    """
    # 创建数据的副本，以避免对原始数据集的修改
    df = data[:]

    # 计算变量的ECDF
    ecdf = ECDF(df[variable])

    # 选择x轴的范围，从最小值到第99.9百分位数
    x = np.linspace(min(df[variable]), np.nanpercentile(df[variable], 99.9))

    # 计算对应x轴值的ECDF的y值
    y = ecdf(x)

    # 使用step函数绘制ECDF图
    plt.step(x, y)



# DebtRatio is distributed Overall
def debitRatio():
    df = pd.read_csv("data/cs-training.csv")
    perc = range(81)
    val = []
    for i in perc:
        val.append(np.percentile(df['DebtRatio'], i))
    plt.plot(perc, val, 'go-', linewidth=2, markersize=12)


def debtRatioAboutIncome():
    """
    显示有收入的人的负债率最高的99.0～99.9的debitRatio曲线
    和收入缺失的人的整体负债率
    :return:他们x轴不同，但曲线增长的对应的y轴值大致相同
    """
    df_mis_inc, df_not_mis_inc, var_names = read_data()
    perc1 = [99.0, 99.1, 99.2, 99.3, 99.4, 99.5, 99.6, 99.7, 99.8, 99.9]
    perc2 = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    val1 = []
    val2 = []
    for i in perc1:
        val1.append(np.percentile(df_not_mis_inc['DebtRatio'], i))# 收入无缺失且负债比率高于99%的人的经验曲线
        #给出来的值是超过i%的值，该值是属于df_not_mis_inc['DebtRatio']的一项值
    for i in perc2:
        val2.append(np.percentile(df_mis_inc['DebtRatio'], i)) # 收入缺失，负债比例从0到90的人的经验曲线
    plt.plot(perc1, val1)
    plt.plot(perc2, val2)
    plt.show()

def NumberOfDependents():
    """
    该函数用于展示两个数据集中"NumberOfDependents"变量的直方图，
    分别是df_not_mis_inc和df_mis_inc
    :return:
    """
    df_mis_inc, df_not_mis_inc, var_names = read_data()
    df_not_mis_inc.hist("NumberOfDependents")
    plt.xticks(np.arange(0,20,1))
    df_mis_inc.hist("NumberOfDependents")
    plt.xticks(np.arange(0,20,1))
    plt.show()

def NumberOfDependents_about_income():
    """
    该函数用于分析不同家庭成员数量下的平均月收入情况
    """
    df_mis_inc, df_not_mis_inc, var_names = read_data()
    print(df_mis_inc['NumberOfDependents'].value_counts()) # 缺失收入的家庭人数的频数统计
    print(df_not_mis_inc.loc[df_not_mis_inc['NumberOfDependents']==0,["MonthlyIncome"]].mean()) # 含有家庭收入的人口为0的平均收入均值
    print(df_not_mis_inc.loc[df_not_mis_inc['NumberOfDependents']==1,['MonthlyIncome']].mean())
    print(df_not_mis_inc.loc[df_not_mis_inc['NumberOfDependents']>1,["MonthlyIncome"]].mean())

if __name__ == '__main__':
    debtRatioAboutIncome()
    NumberOfDependents()
    NumberOfDependents_about_income()



