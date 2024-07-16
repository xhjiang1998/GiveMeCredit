import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
# 关于本函数的数理逻辑链接：https://zhuanlan.zhihu.com/p/34073898
# 计算分位数的方法：https://zhuanlan.zhihu.com/p/235345817

#加载数据
from missing_value_analysis import visualizeECDF


def load_data():
    df=pd.read_csv("data/cs-training.csv")
    cols = df.columns
    return df,cols

# 利用上下分位数计算正常值范围，统计异常特征值的数量
def outlierAnalysis():
    df,df1,cols=load_data()
    np.set_printoptions(suppress=True)
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    for i in cols:
        Low_Bound=((np.percentile(df[i],25))- 1.5*np.subtract(*np.percentile(df[i],[75,25])))
        Upp_Bound = ((np.percentile(df[i], 75)) + 1.5 * np.subtract(*np.percentile(df[i], [75, 25])))
        df1[i] = df[i].apply(lambda x: 1 if (x < Low_Bound or x > Upp_Bound) else 0) # 0就是正常值范围内的
        df2 = df1.sum(axis=0, skipna=True)
        print(df2) #行累加，按照列归纳出结果，跳过为null值的
    print(df2.sort_values());


# 描绘三个连续变量的分布
# 无抵押贷款循环使用率，除不动产和车贷之外的贷款余额与个人信用总额度之比
# 负债比率
# 月收入
def outilerprocess():
    varNames = ['RevolvingUtilizationOfUnsecuredLines', 'DebtRatio', 'MonthlyIncome']
    df,df1,cols=load_data()
    np.random.seed(100)
    fig, axes = plt.subplots(nrows=3, ncols=2)  # 返回包含图像和轴对象的元组的函数
    fig.tight_layout()
    fig.set_figheight(10)
    fig.set_figwidth(15)
    plt.subplots_adjust(hspace=0.5)
    ax = plt.subplot(3,2,1)
    ax.set_title(varNames[0])
    visualizeECDF(varNames[0],df)
    perc = [97.5,98,98.5,99,99.5,100]
    val1 = []
    for i in perc:
        val1.append(np.percentile(df['RevolvingUtilizationOfUnsecuredLines'],i ))
    ax = plt.subplot(3,2,2)
    ax.set_title(varNames[0])
    plt.plot(perc, val1, 'go-', linewidth=2, markersize=12) # perc为x轴，val为y轴
    for a, b in zip(perc,val1):
        plt.text(a, b, (a,b),verticalalignment='top', va='center', fontsize=10)

    ax = plt.subplot(3,2,3)
    ax.set_title(varNames[1])
    visualizeECDF(varNames[1], data = df)
    perc = [97.5,98,98.5,99,99.5,100]
    val2 = []
    for i in perc:
        val2.append(np.percentile(df['DebtRatio'],i ))
    ax = plt.subplot(3,2,4)
    ax.set_title(varNames[1])
    plt.plot(perc, val2,'go-', linewidth=2, markersize=12)
    for a, b in zip(perc,val2):
        plt.text(a, b, (a,b),ha='center', va='bottom', fontsize=10)

    # Full distribution vs top 2.5 percentile distribution
    ax = plt.subplot(3,2,5) # 3行2列第5个
    ax.set_title(varNames[2])
    # visualizeECDF(varNames[2], data = df[df[varNames[2]].notnull()])
    visualizeECDF(varNames[2], data = df) # 描绘MonthlyIncome的经验分布函数
    perc = [97.5,98,98.5,99,99.5,100]
    val3 = []
    for i in perc:
        # val3.append(np.percentile(df['MonthlyIncome'],i )) # 这里对于高小数位会直接取值nan
        val3.append(np.nanpercentile(df1['MonthlyIncome'],i ))
    ax = plt.subplot(3,2,6)
    ax.set_title(varNames[2])
    plt.plot(perc, val3, 'go-', linewidth=2, markersize=12)
    for a, b in zip(perc,val3):
        plt.text(a, b, (a,b),ha='center', va='bottom', fontsize=10)
    plt.show()


if __name__ == '__main__':
    outilerprocess()