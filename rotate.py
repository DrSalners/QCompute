import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from tools import string_operator, operate, expectation_value,controlled_op

# # one qubit
# psi=np.zeros((2))
# psi[1]=1
# z=np.array([[1,0],[0,-1]])
# y=np.array([[0,-1j],[1j,0]])
# x=np.array([[0,1],[1,0]])
# rx_pi=np.array([[np.cos(np.pi/2),-1j*np.sin(np.pi/2)],[-1j*np.sin(np.pi/2),np.cos(np.pi/2)]])
# ry_pi=np.array([[np.cos(np.pi/2),-1*np.sin(np.pi/2)],[np.sin(np.pi/2),np.cos(np.pi/2)]])
# rz_pi=np.array([[np.exp(-1j*np.pi/2),0],[0,np.exp(1j*np.pi/2)]])

# print(z,rz_pi)
# print(np.dot(z,psi))
# print(np.dot(x,psi))
# print(np.dot(y,psi))
# print(np.dot(rx_pi,psi))
# print(np.dot(ry_pi,psi))
# print(np.dot(rz_pi,psi))

# two qubits
n_qubits=2
psi=[0 for i in range(2**n_qubits)]
psi[0]=1

IH=string_operator(['I','H']).generate_matrix()
CX=string_operator(['CX']).generate_matrix()
CX2=string_operator(['CX2']).generate_matrix()
CH=string_operator(['CH']).generate_matrix()
XI=string_operator(['I','Px']).generate_matrix()
IX=string_operator(['Px','I']).generate_matrix()
XH=string_operator(['Px','H']).generate_matrix()
ZZ=string_operator(['Pz','Pz']).generate_matrix()

CX_1_2=controlled_op(1,2,2,'Px').generate_matrix()
CX_2_1=controlled_op(2,1,2,'Px').generate_matrix()


# print(operate(CX2,operate(IX,psi)))



# #Generate bell state 1
# psi=[0 for i in range(2**n_qubits)]
# psi[0]=1
# psi=operate(IH,psi)
# psi=operate(Cx,psi)
# print('bell state 1:',psi)

# #Generate bell state 2
# psi=[0 for i in range(2**n_qubits)]
# psi[0]=1
# psi=operate(XI,psi)
# psi=operate(IH,psi)
# psi=operate(Cx,psi)
# print('bell state 2:',psi)

# #Generate bell state 3
# psi=[0 for i in range(2**n_qubits)]
# psi[0]=1
# psi=operate(XH,psi)
# psi=operate(Cx,psi)
# print('bell state 3:',psi)

# #Generate bell state 4
# psi=[0 for i in range(2**n_qubits)]
# psi[0]=1
# psi=operate(XH,psi)
# psi=operate(ZZ,psi)
# psi=operate(Cx,psi)
# print('bell state 4:',psi)

# three qubits
#Generate W state
n_qubits=3
psi=[0 for i in range(2**n_qubits)]
psi[0]=1
# print(psi)

theta=1.91063324
IIRy=string_operator(['I','I',['Ry',theta]]).generate_matrix()
ICH=string_operator(['CH','I']).generate_matrix()
CHI=string_operator(['I','CH']).generate_matrix()

ICX=string_operator(['I','CX']).generate_matrix()
CXI=string_operator(['CX','I']).generate_matrix()
IIX=string_operator(['Px','I','I']).generate_matrix()
IXI=string_operator(['I','Px','I']).generate_matrix()
XII=string_operator(['I','I','Px']).generate_matrix()

CX_1_2=controlled_op(1,2,3,'Px').generate_matrix()
CH_1_2=controlled_op(1,2,3,'H').generate_matrix()
CX_2_3=controlled_op(2,3,3,'Px').generate_matrix()

# print(operate(XII,psi))
# print(operate(CX_1_2,operate(XII,psi)))


psi1=operate(IIRy,psi)
psi2=operate(CH_1_2,psi1)
psi3=operate(CX_2_3,psi2)
psi4=operate(CX_1_2,psi3)
psi5=operate(XII,psi4)
print(psi5)


# # compute expectation value of some observables
# obs_lab=['IZ','IX','ZI','XI','ZZ','XX']
# observables = [['I','Pz'], ['I','Px'], ['Pz','I'], ['Px','I'], ['Pz','Pz'], ['Px','Px']]
# ev=[]
# for obs in observables:
#     mat=string_operator(obs).generate_matrix()
#     ev.append(expectation_value(mat,psi))

# measurement=[0,0]
# rand=np.random.random(100)
# measurement[0]=sum(rand>wf[0]**2)
# measurement[1]=sum(rand<=wf[0]**2)
# print(measurement)

# mat=string_operator(['I','Px']).generate_matrix
# wf=operate(mat(),psi)
# print(wf)

# mat=string_operator(['I','Py']).generate_matrix
# wf=operate(mat(),psi)
# print(wf)

# mat=string_operator(['I','H']).generate_matrix
# psi=operate(mat(),psi)
# mat=string_operator(['CX']).generate_matrix
# wf=operate(mat(),psi)
# print(wf)

# mat=string_operator(['I','I','Px']).generate_matrix
# mat_1,wf=tools.Px(psi,['I','Px'])

# I=[[1,0],[0,1]]

# mat=[[0 for i in range(8)] for j in range(8)]
# for op in range(len())
#     for i in range(8):
#         for j in range(8):
#             mat[i][j]=mat_1[i-np.mod(i,2)-int(np.floor(i/2))][j-np.mod(j,2)-int(np.floor(j/2))]*I[np.mod(i,2)][np.mod(j,2)]
# print(mat)
#### vizualization
# fig = plt.figure(figsize=(12, 12))
# ax = fig.add_subplot(projection='3d')
# array=np.array([1,1,1])
# array=array/np.linalg.norm(array)o
# ax.set_xlim([-1,1])
# ax.set_ylim([-1,1])
# ax.set_zlim([-1,1])
# for i in range(10):
#     array=np.rot90(array)
#     ax.plot([0,array[0]],[0,array[1]],zs=[0,array[2]])
#     plt.pause(1)
# plt.show()