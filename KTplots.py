import numpy as nm
import matplotlib.pyplot as plt
from matplotlib import cm
import pdb
plt.rcParams["font.size"] = 20
result = nm.load("KTIV2/results.npy",allow_pickle=True).item()
T = result['T_range']
cond = result['cond']
V_range=result['V_range']
TunnelI=result['TunnelI']
fig2,ax2 = plt.subplots()
for j in range(1,10):
    ax2.plot(T,TunnelI[j,:]*1E3,'*-')
ax2.set_xlabel('Temperature (K)')
ax2.set_ylabel('I (mA)')
fig3,ax3 = plt.subplots()
#ax2.plot(V_range,TunnelI[:,2])
for j in range(0,T.size):
    ax3.plot(V_range*1E3,TunnelI[:,j]*1E3,linewidth=2)
ax3.set_xlabel('Potential (mV)')
ax3.set_ylabel('I (mA)')
############# Sensitivity
fig4,ax4 = plt.subplots()
for j in range(1,10):
    Y = nm.diff(nm.log(TunnelI[j,:]))/nm.diff(nm.log(T))
    ax4.plot(T[0:-1],Y,'*-')
ax4.set_xlabel('Temperature (K)')
ax4.set_ylabel('S')
plt.show()
