import pandas as pd
import matplotlib.pyplot as plt


length = pd.read_csv('/home/luokai/data/rebuild_features_HTCYC1132.csv')['cycle_life'][0] # cycle_life

x=[]
y=[]
for j in range (1, length-1):         

    cap = pd.read_csv('/data/base/NFSfile/DATA/TsinghuaWX/HTCYC/HTCYC0102' + '_charge_' + f'{j}' + '.csv')['cap'][0]
    x.append(j)
    y.append(cap)

plt.switch_backend('agg')
plt.plot(y)
plt.savefig('1.png')




for j in range (50, 71):    
    cap2 = pd.read_csv('/data/base/NFSfile/DATA/TsinghuaWX/HTCYC/HTCYC0102' + '_charge_' + f'{j}' + '.csv')
    print(cap2)
    print('----------------------')