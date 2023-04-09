import numpy as nm
import matplotlib.pyplot as plt
from matplotlib import cm
import pdb
fig1,ax1 = plt.subplots()
result = nm.load("BTKIV/results.npy",allow_pickle=True).item()
T = result['T_range']
cond = result['cond']
V_range=result['V_range']
TunnelI=result['TunnelI']
ax1.plot(T,cond[20,:],'*')
fig2,ax2 = plt.subplots()
#ax2.plot(V_range,TunnelI[:,2])
for j in range(1,10):
    ax2.plot(T,TunnelI[j,:])
ax2.set_xlabel('Temperature')
fig3,ax3 = plt.subplots()
#ax2.plot(V_range,TunnelI[:,2])
for j in range(0,T.size,4):
    ax3.plot(V_range,TunnelI[:,j])
ax3.set_xlabel('Potential')
plt.show()
