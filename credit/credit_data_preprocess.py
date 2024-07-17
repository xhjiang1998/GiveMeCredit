import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF


class Test:
    df_not_mis_inc = {}
    df_mis_inc = []
    varNames = []

    # 数据加载与特征返回
    def read_data(self):
        np.set_printoptions(edgeitems=10)
        np.core.arrayprint._line_width = 180
        # 显示为……的列
        pd.set_option('display.max_columns', 20)
        train_data = pd.read_csv("data/cs-training.csv")
        # 判断是否有重复
        train_data.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)


        df_mis_inc = train_data[train_data['MonthlyIncome'].isna()]  # 判断对应的矩阵为空则赋true，size相同
        df_not_mis_inc = train_data[train_data['MonthlyIncome'].notna()]  # 判断收入没丢的
        varNames = ['RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio',
                    'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
                    'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents']
        return df_mis_inc, df_not_mis_inc, varNames

    # 可视化
    def visualizeECDF(self, variable, data):
        df = data[:]  # 复制
        ecdf = ECDF(df[variable])
        x = np.linspace(min(df[variable]), np.percentile(df[variable], 99.9))
        y = ecdf(x)
        plt.step(x, y)

    # 债务数据分布
    def debitRatio(self):
        df = pd.read_csv("data/cs-training.csv")
        perc = range(81)
        perc = [10, 20, 30, 40, 50, 60, 70, 80]
        val = []
        for i in perc:
            val.append(np.percentile(df['DebtRatio'], i))
        plt.plot(perc, val, 'go-', linewidth=2, markersize=12)

    # 债务比关于收入的经验函数分布
    def debtRatioAboutIncome(self):
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

    def showVariable(self):
        # 类中函数调用其他函数
        df_mis_inc, df_not_mis_inc, var_names = Test.read_data(self)
        print(type(df_mis_inc))
        print(df_not_mis_inc)

    def test(self):
        df = pd.DataFrame({'age': [5, 6, np.NaN],
                           'born': [pd.NaT, pd.Timestamp('1939-05-27'), pd.Timestamp('1940-04-25')],
                           'name': ['Alfred', 'Batman', ''],
                           'toy': [None, 'Batmobile', 'Joker']})
        print(df)
        print(df['name'].isna())
        print(df[df['name'].isna()])


if __name__ == '__main__':
    # df_mis_inc,df_not_mis_inc,varNames=Test().read_data()
    # Test().read_data()

    # np.random.seed(100)
    # fig, axes = plt.subplots(nrows=9, ncols=2)
    # fig.tight_layout()
    # fig.set_figheight(45)
    # fig.set_figwidth(15)
    # plt.subplots_adjust(hspace=0.8)
    # for i in [1, 3, 5, 7, 9, 11, 13, 15, 17]:
    #     ax = plt.subplot(9, 2, i)
    #     ax.set_title(varNames[(i - 1) // 2])
    #     Test().visualizeECDF(varNames[(i - 1) // 2], df_not_mis_inc)
    #     ax = plt.subplot(9, 2, i + 1)
    #     ax.set_title(varNames[(i - 1) // 2])
    #     Test().visualizeECDF(varNames[(i - 1) // 2], df_mis_inc)

    # 百分之八十的负债比率都在1.0以下
    # Test().debitRatio()
    # Test().debtRatioAboutIncome()

    # Test().showVariable()

        Test().test()