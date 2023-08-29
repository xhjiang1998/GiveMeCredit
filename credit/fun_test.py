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
x=[1,2,3]
y=[1,2]
plt.plot(x,y,'-',color='blue')
plt.show()
