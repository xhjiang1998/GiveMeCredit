import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF



# 数据加载与特征返回
def read_data():
    np.set_printoptions(edgeitems=10)
    np.core.arrayprint._line_width = 180
    # 显示为……的列
    pd.set_option('display.max_columns', 20)
    train_data = pd.read_csv("cs-training.csv")
    # 判断是否有重复
    train_data.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)


    df_mis_inc = train_data[train_data['MonthlyIncome'].isna()]  # 判断对应的矩阵为空则赋true，size相同
    df_not_mis_inc = train_data[train_data['MonthlyIncome'].notna()]  # 判断收入没丢的
    varNames = ['RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio',
                'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
                'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents']
    return df_mis_inc, df_not_mis_inc, varNames

# 可视化
def visualizeECDF( variable, data):
    df = data[:]  # 复制
    ecdf = ECDF(df[variable])
    x = np.linspace(min(df[variable]), np.percentile(df[variable], 99.9))
    y = ecdf(x)
    plt.step(x, y)



def debtRatioAboutIncome():
    """
    债务比最大的99%～99.9%但有收入的人和0%～90%的无收入的人债务两幅图
    图形横坐标为比例，纵坐标为债务比
    """
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



def monthly_income_with_mul_variable_ECDF():
    """
        月收入值是否缺失在多个变量上的经验累积分布函数，debitRito、numberofdependents曲线不一样
        :return:
    """
    df_mis_inc, df_not_mis_inc, varNames = read_data()
    np.random.seed(100)
    fig, axes = plt.subplots(nrows=5, ncols=4)
    fig.tight_layout()
    fig.set_figheight(30)
    fig.set_figwidth(20)
    plt.subplots_adjust(hspace=0.8,wspace=0.3)
    for i in [1, 3, 5, 7, 9, 11, 13, 15, 17]:
        ax = plt.subplot(9, 2, i)
        ax.set_title(varNames[(i - 1) // 2])
        visualizeECDF(varNames[(i - 1) // 2], df_not_mis_inc)
        ax = plt.subplot(9, 2, i + 1)
        ax.set_title(varNames[(i - 1) // 2])
        visualizeECDF(varNames[(i - 1) // 2], df_mis_inc)

    # for i, varName in enumerate(varNames):  # 使用enumerate简化循环
    #     ax = plt.subplot(5, 4, i + 1)  # 更新子图索引计算方式
    #     ax.set_title(varName)
    #     visualizeECDF(varName, df_not_mis_inc)
    #
    #     ax = plt.subplot(5, 4, i + 5)  # 继续在下一行绘制第二个ECDF
    #     ax.set_title(varName)
    #     visualizeECDF(varName, df_mis_inc)
    filename= "images/monthly_income_ECDF_distribution.png"
    plt.savefig(filename)
    plt.show()
    print(f"ECDF图形已保存为：{filename}")


if __name__ == '__main__':
    monthly_income_with_mul_variable_ECDF()
    debtRatioAboutIncome()
    # 由于两幅图图形类似，所以可以用99%～99.9%的人收入去代替未有收入的人月收入的
    # Test().showVariable()
    # Test().test()