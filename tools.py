import numpy as np

def operate(op: list[list],
            wf: list,
            )-> list:
    """Performs operation on wavefunction.
    
    Args:
    op (list[list]): Matrix representation of operator
    wf (list): Vector representation of wavefunction
    
    Returns:
    wf (list): The new wavefunction
    """

    # intialze new wavfunction to populate
    vec=[0 for i in range(len(op))]
    # perform matrix multiplication
    for i in range(len(op)):
        for j in range(len(op)):
            vec[i]+=op[i][j]*wf[j]

    return vec


def matrix_add(mat1: list[list],
               mat2: list[list],
               )-> list[list]:
    """ Adds two matrices together.
    
    Args:
    mat1 (list[list]): matrix 1
    mat2 (list[list]): matrix 2
    
    Returns:
    mat (list[list]): matrix 1 + matrix 2
    """

    if len(mat1)!=len(mat2):
        ValueError('Matrices are not the same dimensions')
    mat=[[0 for i in range(len(mat1))] for j in range(len(mat1))]
    for i in range(len(mat1)):
        for j in range(len(mat2)):
            mat[i][j]=mat1[i][j]+mat2[i][j]
    return mat

def matrix_subtract(mat1: list[list],
               mat2: list[list],
               )-> list[list]:
    """ Subtracts two matrices (mat1-mat2).
    
    Args:
    mat1 (list[list]): matrix 1
    mat2 (list[list]): matrix 2
    
    Returns:
    mat (list[list]): matrix 1 - matrix 2
    """

    if len(mat1)!=len(mat2):
        ValueError('Matrices are not the same dimensions')
    mat=[[0 for i in range(len(mat1))] for j in range(len(mat1))]
    for i in range(len(mat1)):
        for j in range(len(mat2)):
            mat[i][j]=mat1[i][j]-mat2[i][j]
    return mat

def matrix_multiply(mat1: list[list],
                    mat2: list[list],
                    )-> list[list]:
    """ Multiplies two matrices together.
    
    Args:
    mat1 (list[list]): matrix 1
    mat2 (list[list]): matrix 2
    
    Returns:
    mat (list[list]): matrix 1 * matrix 2
    """
    
    if len(mat1[0])!=len(mat2):
        ValueError('len of mat1 column n.e. to len of mat2 row')

    mat=[[0 for i in range(len(mat1[0]))] for j in range(len(mat1[0]))]
    for i in range(len(mat1[0])):
        for j in range(len(mat1[0])):
            for k,l in zip(range(len(mat1[0])),range(len(mat1[0]))):
                mat[i][j]+=mat1[i][k]*mat2[l][j]
    return mat

def expectation_value(op: list[list],
                      wf: list,
                      )-> float:
    """Computes the expectation value of an observable in the given wavefunction.
    
    Args:
    op (list[list]): Matrix representation of operator
    wf: (list): Vector representation of the wavefunction
    
    Return:
    expected_value (float): The expected value of the observable
    """

    expected_value=[wf[i].conjugate()*operate(op,wf)[i] for i in range(len(wf))]
    return sum(expected_value)

def exact_evolve(psi: list,
                 hamiltonian: str,
                 evolution_time: int|float,
                 time_step: int|float,
                 )-> list[list]:
    """Generates exact time evolution of single qubit psi based on hamiltonian = Px, Py or Pz.
    
    Args:
        psi (list): vector rep. of input wavefunction
        hamiltonian (list[list]): matrix rep. of Hamiltonian
        evolution_time (int|float): total time of evolution
        time_step (int|float): time step
        
    Returns (list[list]): trajectory of psi over time evolution
    """
    U=string_operator([['R'+hamiltonian[-1],-2*time_step]]).generate_matrix()
    trajectory=[]
    time=np.arange(0,evolution_time,time_step)
    for t in time:
        psi=operate(U,psi)
        trajectory.append(psi)
    return trajectory

def stag_mag(psi: list,
             n: int,
             )-> float:
    """Computes Staggered Magnetization operaterator
    
    Args:
        psi (list): vector rep. of wavefunction
        n (int): number of qubits
        
    Returns (float): value of staggered magnetization
    """
    val=0
    for i in range(n):
        op_string=[0 for j in range(n)]
        for j in range(n):
            if i==j:
                op_string[j]='Pz'
            else:
                op_string[j]='I'
        val+=(((-1)**(i+1))*expectation_value(string_operator(op_string).generate_matrix(),psi))/n
    return val

def tensor_prod(dim: int,
                mat1: list[list],
                mat2: list[list],
                )-> list[list]:
    l=len(mat1)
    mat=[[0 for i in range(dim)] for j in range(dim)]
    for i in range(dim):
        for j in range(dim):
            mat[i][j]=mat2[i-np.mod(i,l)-(l-1)*int(np.floor(i/l))][j-np.mod(j,l)-(l-1)*int(np.floor(j/l))]*mat1[np.mod(i,l)][np.mod(j,l)]
    return mat

def Heis_Ham_1D_Propogator(n_qubit: int,
                           n: int,
                           time: float|int,
                           J_vec: list[float],
                           )-> list[list]:
    """Constructs matrix representation of one trotter step for a heisenburg propogator, U_heis.
    
    Args:
        n_qubit(int): number of qubits
        n(int): number of trotter slices
        time(float|int): time to simulate
        
    Returns
        U_hies(list[list]): Matrix representation of propogator
    """
    # define parameters
    deltat=time/n
    print(deltat)
    # construct Cx,H,Y,Ydag and Rz
    Cx=controlled_op(1,2,2,'Px').generate_matrix()
    H=string_operator(['H','H']).generate_matrix()
    Y=string_operator(['Y','Y']).generate_matrix()
    Ydag=string_operator(['Ydag','Ydag']).generate_matrix()
    Rz=string_operator([['Rz',J_vec[2]*-2*deltat],'I']).generate_matrix()
    Rz_x=string_operator([['Rz',J_vec[0]*-2*deltat],'I']).generate_matrix()
    Rz_y=string_operator([['Rz',J_vec[1]*-2*deltat],'I']).generate_matrix()

    list=[Cx,Rz_x,Cx,H,Y,Cx,Rz_y,Cx,Ydag,Cx,Rz,Cx]

    #contstruct even operator
    for i in [2*a for a in range(int(n_qubit/2))]:
        U_even_single=H.copy()
        for mat in list:
            U_even_single=matrix_multiply(mat,U_even_single)
        if i==0:
            U_even=U_even_single.copy()
        else:
            U_even=tensor_prod(len(U_even)*len(U_even_single),U_even_single,U_even)
    
    #construct U_odd
    if n_qubit>2:
        for i in [2*a+1 for a in range(int(np.ceil(n_qubit/2.0))-1)]:
            U_odd_single=H.copy()
            for mat in list:
                U_odd_single=matrix_multiply(mat,U_odd_single)
            if i==1:
                U_odd=U_odd_single.copy()
            else:
                U_odd=tensor_prod(len(U_odd)*len(U_odd_single),U_odd_single,U_odd)
        
        # multiply U_even with U_odd
        I=string_operator(['I']).generate_matrix()
        if n_qubit%2==0:
            U_odd=tensor_prod(len(I)*len(U_odd),I,U_odd)
            U_odd=tensor_prod(len(I)*len(U_odd),U_odd,I)
        else:
            U_odd=tensor_prod(len(I)*len(U_odd),I,U_odd)
            U_even=tensor_prod(len(I)*len(U_even),U_even,I)
        U=matrix_multiply(U_odd,U_even) 

    else:
        U=U_even.copy()
        
    return U

def state_prod(state1, state2):
    state=[]
    for j in range(len(state2)):
        for i in range(len(state1)):
            state.append(state1[i]*state2[j])
    return state


def generate_neel(n):
    neel_st=[1,0]
    for i in range(n-1):
        if np.mod(i,2)==1:
            neel_st=state_prod(neel_st,[1,0])
        else:
            neel_st=state_prod(neel_st,[0,1])
    return neel_st


class string_operator ():
    # define matrix dictionary
    mat_dict={
        '|0><0|':[[1,0],
                  [0,0]],
        '|1><1|':[[0,0],
                  [0,1]],
        'I':[[1,0],
             [0,1]],
        'Px':[[0,1],
              [1,0]],
        'Py':[[0,-1j],
              [1j,0]],
        'Pz':[[1,0],
              [0,-1]],
        'H':[[1/np.sqrt(2),1/np.sqrt(2)],
             [1/np.sqrt(2),-1/np.sqrt(2)]],
        'Y':[[1.0/2.0+1j/2.0, 1.0/2.0-1j/2.0],
             [1.0/2.0-1j/2.0, 1.0/2.0+1j/2.0]],
        'Ydag':[[1.0/2.0-1j/2.0, 1.0/2.0+1j/2.0],
                [1.0/2.0+1j/2.0, 1.0/2.0-1j/2.0]],
        'S':[[1,0],
             [0,1j]],
        'T':[[1,0],
             [0,np.exp(1j*np.pi/4)]],
        'CX':[[1,0,0,0],
              [0,1,0,0],
              [0,0,0,1],
              [0,0,1,0]],
        'CX2':[[1,0,0,0],
               [0,0,0,1],
               [0,0,1,0],
               [0,1,0,0]],
        'CZ':[[1,0,0,0],
              [0,1,0,0],
              [0,0,1,0],
              [0,0,0,-1]],
        'CH':[[1,0,0,0],
              [0,1,0,0],
              [0,0,1.0/np.sqrt(2),1.0/np.sqrt(2)],
              [0,0,1.0/np.sqrt(2),-1.0/np.sqrt(2)]],
        'Swap':[[1,0,0,0],
                [0,0,1,0],
                [0,1,0,0],
                [0,0,0,1]],
        'Tof':[[1,0,0,0,0,0,0,0],
               [0,1,0,0,0,0,0,0],
               [0,0,1,0,0,0,0,0],
               [0,0,0,1,0,0,0,0],
               [0,0,0,0,1,0,0,0],
               [0,0,0,0,0,1,0,0],
               [0,0,0,0,0,0,0,1],
               [0,0,0,0,0,0,1,0]],
            }
    mat_dict['Rx']=lambda theta: [[np.cos(theta/2.0),-1j*np.sin(theta/2.0)],
                                    [-1j*np.sin(theta/2.0),np.cos(theta/2.0)]]
    mat_dict['Ry'] = lambda theta: [[np.cos(theta/2.0),-1*np.sin(theta/2.0)],
                                    [np.sin(theta/2.0),np.cos(theta/2.0)]]
    mat_dict['Rz'] = lambda theta:[[np.exp(-1j*theta/2.0),0],
                                    [0,np.exp(1j*theta/2.0)]]
    def __init__(self,op_string):
        self.op_string=op_string

        if len(self.op_string)!=1:
            if type(op_string[len(self.op_string)-1])==list:
                self.temp_mat=self.mat_dict[op_string[len(self.op_string)-1][0]](op_string[len(self.op_string)-1][1])
            else:
                self.temp_mat=self.mat_dict[op_string[len(self.op_string)-1]]

        self.n_ops=len(op_string)

        
    def generate_matrix(self,
                        ) -> list[list]:
        """Generates matrix representation of tensor products of strings of operators:
        
        Args:
        self (obj): and instance of the string_operator object

        Returns (list[list]): Matrix representation of tensor product of operators
        """
        if self.n_ops==1:
            if type (self.op_string[0])==list:
                mat=self.mat_dict[self.op_string[0][0]](self.op_string[0][1])
            else:
                mat=self.mat_dict[self.op_string[0]]

        else:
            # construct n x n matrix operator
            for n in range(2,self.n_ops+1):
                if type (self.op_string[self.n_ops-n])==list:
                    len_1=len(self.temp_mat)
                    len_2=len(self.mat_dict[self.op_string[self.n_ops-n][0]](self.op_string[self.n_ops-n][1]))
                    dim=len_1*len_2
                    mat=tensor_prod(dim,self.mat_dict[self.op_string[self.n_ops-n][0]](self.op_string[self.n_ops-n][1]),self.temp_mat)

                else:
                    len_1=len(self.temp_mat)
                    len_2=len(self.mat_dict[self.op_string[self.n_ops-n]])
                    dim=len_1*len_2
                    mat=tensor_prod(dim,self.mat_dict[self.op_string[self.n_ops-n]],self.temp_mat)
                self.temp_mat=mat.copy()

        return mat

class controlled_op():

    def __init__(self,c_qub,t_qub,n_qub,op):
        self.control=c_qub
        self.target=t_qub
        self.num_qubits=n_qub
        self.operator=op
    
    def generate_matrix(self,
                        )-> list[list]:
        """ Generates matrix representation of controlled operation.
        
        Args:
        self
        
        Returns:
        mat(list[list]): Matrix representation of controlled operation"""

        #generate matrix 1
        op_string=['I' for i in range(self.num_qubits)]
        op_string[self.control-1]='|0><0|' 
        mat1=string_operator(op_string[::-1]).generate_matrix()

        #generate matrix 2
        op_string=['I' for i in range(self.num_qubits)]
        op_string[self.control-1]='|1><1|'
        op_string[self.target-1]=self.operator
        mat2=string_operator(op_string[::-1]).generate_matrix()

        return matrix_add(mat1,mat2)