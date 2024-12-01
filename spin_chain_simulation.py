import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import tools
from tools import string_operator
import time as tm

dt=2.5
n=100
time=np.arange(0,dt,dt/n)
numqubits=[3]
Jx=-0.8
Jy=-0.2
Jz=0
sim_time=[]
for nqubits in numqubits:
    time_start=tm.time()
    neel=tools.generate_neel(nqubits)
    trajectory=[]
    U_2=tools.Heis_Ham_1D_Propogator(nqubits,n,dt,[Jx,Jy,Jz])
    psi=neel.copy()
    for t in time:
        if t==0:
            trajectory.append(psi)
            continue
        psi=tools.operate(U_2,psi)
        trajectory.append(psi)
    sm=[]
    for state in trajectory:
        sm.append(tools.stag_mag(state,nqubits))
    plt.subplot(1,2,1)
    plt.plot(time,sm,'-*')
    plt.subplot(1,2,2)
    sim_time.append(tm.time()-time_start)
    print(tm.time()-time_start)
res=np.polyfit(numqubits,np.log10(sim_time),1)
plt.plot(numqubits,np.log10(sim_time),'-o')
numqubits.append(20)
print(res)
plt.plot(numqubits,res[1]+np.multiply(numqubits,res[0]),'--k')
plt.show()


