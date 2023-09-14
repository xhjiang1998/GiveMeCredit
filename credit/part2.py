#Importing all the necessary packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
def warn(*args, **kwargs): pass
import warnings
warnings.warn = warn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd # for data analytics
import numpy as np # for numerical computation
from matplotlib import pyplot as plt, style # for ploting
import seaborn as sns # for ploting
from sklearn.metrics import fbeta_score, make_scorer, precision_score, recall_score, confusion_matrix # for evaluation
import itertools
np.random.seed(42)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def std():
    df=pd.read_csv("cs-training.csv")
    sclar=StandardScaler()
    df_std=pd.DataFrame(sclar.fit_transform(df.drop(['Unnamed: 0','SeriousDlqin2yrs'],axis=1)),
    columns=df.drop(['Unnamed: 0','SeriousDlqin2yrs'],axis=1).columns)
    print(df_std)
    pca=PCA()
    pca.fit(df_std)
    print(pca.explained_variance_ratio_.cumsum())

if __name__ == '__main__':
    std()