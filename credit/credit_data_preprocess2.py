import numpy as np
import pandas as pd
from OutlierAnalysis import load_data
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
    df = pd.read_csv("cs-training.csv")
    df_mis_inc = df[df['MonthlyIncome'].isna()]  # 只输出为true的df，而true就是MonthlyIncome为空的
    df_not_mis_inc = df[df['MonthlyIncome'].notna()]  # 输出收入没丢的
    varNames = ['RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio',
                'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
                'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents']
    return df_mis_inc, df_not_mis_inc, varNames

def visual_data():
    df=read_data()
    print('df.head()','\n',df.head())
    print('df.columns','\n',df.columns)
    print('df.shape','\n',df.shape)
    print('df.dtypes','\n',df.dtypes)
    df.SeriousDlqin2yrs=(df.SeriousDlqin2yrs).astype('category')
    df.describe()
    df.info()
    print('df.describe()','\n',df.describe)
    print('df.info()','\n',df.info)
    percent_miss_MonthlyIncome=np.round(sum(df['MonthlyIncome'].isna())/df.shape[0],2)
    percent_miss_NumberOfDependents=np.round(sum(df['NumberOfDependents'].isna())/df.shape[0],2)
    print('percent_miss_MonthlyIncome:{}'.format(percent_miss_MonthlyIncome))
    print('percent_miss_NumberOfDependents:{}'.format(percent_miss_NumberOfDependents))


# 计算部分特征对应的特征值
def adjust_value_count():
    df,df1,cols=load_data()
    df_mis_inc, df_not_mis_inc, varNames=read_data()
    print(df['NumberOfTime30-59DaysPastDueNotWorse'].value_counts())
    print(df['NumberOfTimes90DaysLate'].value_counts())
    print(df['NumberOfTime60-89DaysPastDueNotWorse'].value_counts())

    # 将部分明显的偏差正常的值纠正
    df.loc[df['NumberOfTimes90DaysLate']>20,'NumberOfTimes90DaysLate']=20
    df.loc[df['NumberOfTime60-89DaysPastDueNotWorse']>20,'NumberOfTime60-89DaysPastDueNotWorse']=20
    df.loc[df['NumberOfTime30-59DaysPastDueNotWorse']>20,'NumberOfTime30-59DaysPastDueNotWorse']=20
    # 观察结果(过去两年中发生60-89天逾期的次数)
    print(df['NumberOfTime60-89DaysPastDueNotWorse'].value_counts())

    # 收入缺失值填补
    # df.loc[df[df['MonthlyIncome'].isna()],'MonthlyIncome']=0 错误写法
    df['MonthlyIncome']=df['MonthlyIncome'].fillna(0)
    print(df[df['MonthlyIncome'].isna()].size)

    # 亲属缺失值使用中位数来填充
    df[df['NumberOfDependents'].isna()].describe()
    df[df['NumberOfDependents'].notna()].describe()


if __name__ == '__main__':
    adjust_value_count()


