import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import tools
from tools import string_operator, operate, expectation_value,controlled_op


### exact evolutionn
# trajectory=tools.exact_evolve(psi,'Pz',10,0.1)
# val_x=[]
# val_y=[]
# val_z=[]
# for state in trajectory:
#     val_x.append(tools.expectation_value(string_operator(['Px']).generate_matrix(),state))
#     val_y.append(tools.expectation_value(string_operator(['Py']).generate_matrix(),state))
#     val_z.append(tools.expectation_value(string_operator(['Pz']).generate_matrix(),state))
# plt.plot(val_x)
# plt.plot(val_z)
# plt.plot(val_y)
# plt.show()

### trotterizaiton error test
for n,color,line in zip([1,20],[['red','blue','green'],['red','blue','green']],['-','--']):
    evolution_time=2.5
    evolution_step=evolution_time/50
    sim_steps=np.arange(0,evolution_time,evolution_step)
    trajectory=[]
    for t in sim_steps:
        psi=[1,0]
        if t==0:
            trajectory.append(psi)
            continue
        U1=string_operator([['Rx',2*t/n]]).generate_matrix()
        U2=string_operator([['Rz',2*t/n]]).generate_matrix()
        for step in range(n):
            psi=operate(U1,psi)
            psi=operate(U2,psi)
        trajectory.append(psi)

    val_x=[]
    val_y=[]
    val_z=[]
    for state in trajectory:
        val_x.append(tools.expectation_value(string_operator(['Px']).generate_matrix(),state))
        val_y.append(tools.expectation_value(string_operator(['Py']).generate_matrix(),state))
        val_z.append(tools.expectation_value(string_operator(['Pz']).generate_matrix(),state))
    plt.plot(sim_steps,val_x,color=color[0],linestyle=line)
    plt.plot(sim_steps,val_y,color=color[1],linestyle=line)
    plt.plot(sim_steps,val_z,color=color[2],linestyle=line)
plt.show()
