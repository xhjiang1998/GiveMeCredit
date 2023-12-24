# %%
import pandas as pd
import numpy as np
from credit.credit_data_preprocess2 import visualizeECDF

# %%
a=[1,2,3,4]
b=a[:]
a=[1,2,3]
c={3,2,1}
# c1=c[:]
d=(1,22)
d1=d[:]
print(type(b))
print(type(c))
# print(type(c),c1)
print(type(d),d1)

# print(c[1])
print(d[1])

# %%
import matplotlib.pyplot as plt
fig,[[ax1,ax2],[ax3,ax4]]=plt.subplots(nrows=2,ncols=2)
# plt.show()

# %%
df = pd.read_csv("cs-training.csv")
df_25=np.percentile(df['age'],25)
df_75=np.percentile(df['age'],75)
sub=np.subtract(df_25,df_75)
print(25-1.5*np.subtract(*np.percentile(df['age'], [75, 25])))

# %%
a=1e+3
print(a)

# %%
x=[1,2,3]
y=[1,2,3]
plt.plot(x,y,'-',color='blue')
plt.show()

# %%
df=pd.read_csv("cs-training.csv")
perc = [10,20,30,40,50,60]
for i in perc:
    print(np.nanpercentile(df['MonthlyIncome'],i ))


# %%
df = pd.DataFrame({'age': [5, 6, np.NaN],
                   'born': [pd.NaT, pd.Timestamp('1939-05-27'), pd.Timestamp('1940-04-25')],
                   'name': ['Alfred', 'Batman', ''],
                   'toy': [None, 'Batmobile', 'Joker']})
print(df)
print(df['born'].isna())
print(df[df['born'].isna()])


# %%
import pandas as pd
import numpy as np
df=pd.read_csv("cs-training.csv")
perc = [99.1,99.2,99.5,99.6,99.8,99.9]
val1 = []
for i in perc:
    val1.append(np.percentile(df['RevolvingUtilizationOfUnsecuredLines'],i ))
print(max(df['RevolvingUtilizationOfUnsecuredLines']))
print(val1)
val2=df[df['RevolvingUtilizationOfUnsecuredLines']>10000]
print(val2['RevolvingUtilizationOfUnsecuredLines'].size)

# %%
a=[1,2,3,4,5,6]
b=[1,2,3]
c=[1,3,6]
zipped=zip(a,b,c)
print(list(zipped))


# %%
import numpy as np
clients=[1,2,3,4,5,6]
current_num_join_clients=[1,2,4]
list(np.random.choice(clients, current_num_join_clients, replace=False))
list