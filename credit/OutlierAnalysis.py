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


def outlier_Analysis():
    """
    对数据集进行异常值分析处理。

    本函数加载数据后，计算每列数据的下限和上限，用于识别异常值。
    异常值被定义为落在第25百分位数减去1.5倍的四分位距（IQR）与第75百分位数加上1.5倍的IQR之外的值。
    对于每列数据，标记出异常值，并统计每列数据中的异常值数量。

    返回:
    无直接返回值，但打印了每列数据中异常值的数量以及按异常值数量排序的列名。
    """
    df,cols=load_data()
    df1=df.copy()

    # 设置numpy打印选项，不显示科学计数法
    np.set_printoptions(suppress=True)
    # 设置pandas打印选项，显示浮点数格式为两位小数
    pd.set_option('display.float_format', lambda x: '%.2f' % x)

    # 遍历每一列数据，进行异常值检测
    for i in cols:
        # 计算每列数据的下限，使用四分位数规则定义异常值范围
        Low_Bound=((np.percentile(df[i],25))- 1.5*np.subtract(*np.percentile(df[i],[75,25])))
        # 计算每列数据的上限，使用四分位数规则定义异常值范围
        Up_Bound = ((np.percentile(df[i], 75)) + 1.5 * np.subtract(*np.percentile(df[i], [75, 25])))
        # 标记异常值，如果值落在定义的范围之外，则标记为1，否则为0
        df1[i] = df[i].apply(lambda x: 1 if (x < Low_Bound or x > Up_Bound) else 0) # 0就是正常值范围内的
        # 统计每列数据中的异常值数量
        df2 = df1.sum(axis=0, skipna=True)
        normal_count = df1[i].count()
        # 打印每列数据的异常值数量
        print(f"正常值：{normal_count},异常值:{i}")
        print(df2[i])
    diff_df=df.compare(df1)
    print(diff_df)
    # 打印按异常值数量排序的列名和异常值数量
    print(df2.sort_values())

    # 这段代码在遍历每列数据计算异常值的同时，为每列数据绘制了一个箱线图。箱线图中，红色虚线表示根据四分位数规则确定的上下界，红色散点表示识别出的异常值。
    # 请注意，根据实际情况调整figsize参数以适应你的数据列数和显示需求。此外，根据需要，你也可以选择性地添加小提琴图或其他类型的图表来展示数据分布。

    np.set_printoptions(suppress=True)
    pd.set_option('display.float_format', lambda x: '%.2f' % x)

    fig, axs = plt.subplots(nrows=len(cols), figsize=(10, 5 * len(cols)), squeeze=False)
    for idx, i in enumerate(cols):
        Low_Bound = ((np.percentile(df[i], 25)) - 1.5 * np.subtract(*np.percentile(df[i], [75, 25])))
        Up_Bound = ((np.percentile(df[i], 75)) + 1.5 * np.subtract(*np.percentile(df[i], [75, 25])))
        df1[i] = df[i].apply(lambda x: 1 if (x < Low_Bound or x > Up_Bound) else 0)

        # 绘制箱线图
        axs[idx, 0].boxplot(df[i], vert=False)
        axs[idx, 0].set_title(f'{i} Outliers')
        axs[idx, 0].set_xlabel('Value')
        axs[idx, 0].axvline(Low_Bound, color='r', linestyle='--')  # 绘制下限线
        axs[idx, 0].axvline(Up_Bound, color='r', linestyle='--')  # 绘制上限线

        # 标记异常值点（可选）
        outliers = df[df[i].apply(lambda x: x < Low_Bound or x > Up_Bound)]
        axs[idx, 0].scatter(outliers[i], [idx] * len(outliers), color='red', zorder=5)
    plt.tight_layout()
    plt.savefig('images/outlier_Analysis.png')
    plt.show()

def outiler_process():
    """
    处理并可视化数据集中选定变量的异常值。
    该函数通过绘制经验分布函数（ECDF）和百分位数图，帮助识别和处理数据集中的异常值。
    :return: 无返回值，但生成包含异常值处理结果的图形文件。
    """
    # 定义待处理的变量名称
    varNames = ['RevolvingUtilizationOfUnsecuredLines', 'DebtRatio', 'MonthlyIncome']
    # 加载数据，并对月收入字段进行缺失值处理
    df,cols=load_data()
    # 设置随机数种子，确保结果可复现
    np.random.seed(100)
    # 创建一个包含6个子图的图形，用于绘制ECDF和百分位数图
    fig, axes = plt.subplots(nrows=3, ncols=2)
    # 调整图形布局，避免子图之间间距过小
    fig.tight_layout()
    # 调整图形的高度和宽度
    fig.set_figheight(10)
    fig.set_figwidth(15)
    # 调整子图之间的垂直间距
    plt.subplots_adjust(hspace=0.5)

    # 循环处理每个变量
    for i, varName in enumerate(varNames):
        # 绘制变量的ECDF图
        ax = plt.subplot(3, 2, i * 2 + 1)
        ax.set_title(varName)
        visualizeECDF(varName, df)
        # 计算变量的百分位数，并绘制百分位数图
        perc = [97.5, 98, 98.5, 99, 99.5, 100]
        # 防止因MonthlyIncome没有导致val值为NaN或无穷大，无法绘图posx and posy should be finite values
        if varName == 'MonthlyIncome':
            val = np.nanpercentile(df[varName], perc)
        else:
            val = np.percentile(df[varName], perc)
        ax = plt.subplot(3, 2, i * 2 + 2)
        ax.set_title(varName)
        plt.plot(perc, val, 'go-', linewidth=2, markersize=12)
        # 在百分位数图上添加文本标签
        for a, b in zip(perc, val):
            plt.text(a, b, (a, b), verticalalignment='top' if i < 2 else 'bottom', fontsize=10)

    # 保存图形到文件
    filename = "images/outlier_process.png"
    plt.savefig(filename)
    # 显示图形
    plt.show()

def outlier_process():
    """
    缺失值填补MonthlyIncome、NumberOfDependents
    :return:
    """
    # 加载数据，其中df和df1是数据框，cols是列名列表
    df,df1,cols=load_data()
    # 从数据中读取特定变量，用于后续处理
    df_mis_inc, df_not_mis_inc, varNames=read_data()

    # 输出指定列的值计数，用于了解数据分布情况
    print(df['NumberOfTime30-59DaysPastDueNotWorse'].value_counts())
    print(df['NumberOfTimes90DaysLate'].value_counts())
    print(df['NumberOfTime60-89DaysPastDueNotWorse'].value_counts())

    # 对指定列的值进行限制，将大于20的值设置为20，以处理异常值
    # 将部分明显的偏差正常的值纠正
    df.loc[df['NumberOfTimes90DaysLate']>20,'NumberOfTimes90DaysLate']=20
    df.loc[df['NumberOfTime60-89DaysPastDueNotWorse']>20,'NumberOfTime60-89DaysPastDueNotWorse']=20
    df.loc[df['NumberOfTime30-59DaysPastDueNotWorse']>20,'NumberOfTime30-59DaysPastDueNotWorse']=20
    # 输出处理后的值计数，用于验证处理效果
    # 观察结果(过去两年中发生60-89天逾期的次数)
    print(df['NumberOfTime60-89DaysPastDueNotWorse'].value_counts())

    # 将缺失的月收入值填充为0
    # 收入缺失值填补
    df['MonthlyIncome']=df['MonthlyIncome'].fillna(0)
    # 输出月收入缺失值的数量，用于了解数据质量
    print(df[df['MonthlyIncome'].isna()].size)

    # 对于依赖人数这一列，分别对有值和无值的数据进行描述性统计，用于分析缺失值的处理策略
    # 亲属缺失值使用中位数来填充
    df[df['NumberOfDependents'].isna()].describe()
    df[df['NumberOfDependents'].notna()].describe()


if __name__ == '__main__':
    outiler_process()
    outlier_Analysis()