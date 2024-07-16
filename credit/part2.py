#Importing all the necessary packages
def warn(*args, **kwargs): pass
import warnings
warnings.warn = warn
import pandas as pd # for data analytics
import numpy as np # for numerical computation

np.random.seed(42)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def std():
    """
    主成分分析，在此之前需要手动填补缺失值和Nan
    :return:
    """
    df=pd.read_csv("data/cs-training.csv")
    sclar=StandardScaler()
    df_std=pd.DataFrame(sclar.fit_transform(df.drop(['Unnamed: 0','SeriousDlqin2yrs'],axis=1)),
    columns=df.drop(['Unnamed: 0','SeriousDlqin2yrs'],axis=1).columns)
    print(df_std)
    pca=PCA()
    pca.fit(df_std)
    print(pca.explained_variance_ratio_.cumsum())

if __name__ == '__main__':
    std()