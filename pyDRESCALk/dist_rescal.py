# @author: Namita Kharat,Manish Bhattarai
from numpy import matlib
from . import config

from .utils import *


class rescal_algorithms_2D():
    """
    Performs the distributed RESCAL operation along 2D cartesian grid

    Parameters:
        X_ijk (ndarray) : Distributed Data
        A_ij (ndarray) : Distributed factor A
        R_ijk (ndarray) : Distributed factor R
        params (class): Class which comprises following attributes
        params.comm1 (object): Global Communicator
        params.comm (object): Modified communicator object
        params.k (int) : Rank for decomposition
        params.m (int) : Global dimensions m
        params.n (int) : Global dimensions n
        params.p_r  (int): Cartesian grid row count
        params.p_c  (int): Cartesian grid column count
        params.row_comm (object) : Sub communicator along row
        params.col_comm (object) : Sub communicator along columns
        params.W_update (bool) : flag to set W update True/False
        params.norm (str): NMF norm to be minimized
        params.method(str): NMF optimization method
        params.eps (float) : Epsilon value


    """
    @comm_timing()
    def __init__(self, X_ijk, A_i, A_j, R_ijk, params=None):
        self.params = params
        self.m, self.n, self.p_r, self.p_c, self.k = self.params.m, self.params.n, self.params.p_r, self.params.p_c, self.params.k
        self.comm1 = self.params.comm1  # ['comm1']
        self.cartesian1d_row, self.cartesian1d_column, self.comm = self.params.row_comm, self.params.col_comm, self.params.comm
        self.X_ijk, self.A_i, self.A_j, self.R_ijk = X_ijk, A_i, A_j, R_ijk
        #if self.comm1.rank==0: print(X_ijk.shape,self.A_i.shape,self.A_j.shape,self.R_ijk.shape)
        self.eps = self.params.eps
        self.p = self.p_r * self.p_c
        self.A_update = self.params.A_update
        self.norm = self.params.norm
        self.method = self.params.method
        self.rank = self.comm1.rank
        self.local_A_n = self.A_i.shape[0]
        self.local_R_m = self.R_ijk.shape[0]
        self.np = self.params.np

    def update(self):
        """Performs 1 step Update for factors W and H based on NMF method and corresponding norm minimization

        Returns
        -------
        W_ij : ndarray
           The m/p X k distributed factor W
        H_ij : ndarray
           The k X n/p distributed factor H
        """
        if self.norm.upper() == 'FRO':
            if self.method.upper() == 'MU':
                self.Fro_MU_update(self.A_update)
            else:
                raise Exception('Not a valid method: Choose (mu)')
        else:
            raise Exception('Not a valid norm: Choose (fro)')
        return self.A_i, self.A_j, self.R_ijk

    @comm_timing()
    def row_reduce(self,A):
        """Performs all reduce along row sub communicator"""
        A_TA_glob = self.cartesian1d_row.allreduce(A, op=MPI.SUM)
        self.cartesian1d_row.barrier()
        return A_TA_glob

    @comm_timing()
    def column_reduce(self,A):
        """Performs all reduce along column sub communicator"""
        A_TA_glob = self.cartesian1d_column.allreduce(A, op=MPI.SUM)
        self.cartesian1d_column.barrier()
        return A_TA_glob
 
    @comm_timing()
    def row_broadcast(self,A):
        """Performs broadcast along row sub communicator"""
        A_broadcast = self.cartesian1d_row.bcast(A, root= self.cartesian1d_column.Get_rank())
        self.cartesian1d_row.barrier()
        return A_broadcast

    @comm_timing()
    def column_broadcast(self,A):
        """Performs all reduce along column sub communicator"""
        A_column_broadcast = self.cartesian1d_column.bcast(A, root= self.cartesian1d_row.Get_rank())
        self.cartesian1d_column.barrier()
        return A_column_broadcast

    @count_memory()
    @count_flops()
    @comm_timing()
    def matrix_mul(self,A,B):
        """Computes the matrix multiplication of matrix A and B"""
        AB_local = A@B
        return AB_local
    
    @count_memory()
    @count_flops()
    @comm_timing()
    def gram_mul(self,A):
        """Computes the gram operation of matrix A"""
        A_TA_local = A.T@A
        return A_TA_local


    @comm_timing()
    def global_gram(self, A):

        r""" Distributed gram computation

        Computes the global gram operation of matrix A
        .. math:: A^TA

        Parameters
        ----------
        A  :  ndarray


        Returns
        -------

        A_TA_glob  : ndarray
        """

        A_TA_loc = self.gram_mul(A)
        A_TA_glob = self.row_reduce(A_TA_loc)
        return A_TA_glob

    @comm_timing()
    def row_mm(self, A, B):

        r""" Distributed matrix multiplication along row of matrix

        Computes the matrix multiplication of matrix A and B along row sub communicator
        .. math:: AB

        Parameters
        ----------
        A  :  ndarray
        B  :  ndarray

        Returns
        -------

        AB_glob  : ndarray
        """
        AB_loc = self.matrix_mul(A,B)
        AB_glob = self.row_reduce(AB_loc)
        return AB_glob
    
    @comm_timing()
    def column_mm(self, A, B):
        r""" Distributed matrix multiplication along column of matrix

        Computes the matrix multiplication of matrix A and B along column sub communicator
        .. math:: AB

        Parameters
        ----------
        A  :  ndarray
        B  :  ndarray

        Returns
        -------

        AB_glob  : ndarray
        """
        AB_loc = self.matrix_mul(A,B)
        AB_glob = self.column_reduce(AB_loc)
        return AB_glob
    
    @count_memory()
    @count_flops()
    def element_op(self, A, B, operation):
        """Performs Element operations between A and B"""
        if operation == "mul":
            return A * B
        else:
            return A/B

    def Fro_MU_update(self, A_update=True):
        r"""
        Frobenius norm based multiplicative update of A and R parameter
        Function computes updated A and R parameter for each mpi rank

        Parameters
        ----------
        self : object

        Returns
        -------
        self.A_i : ndarray
        self.R_ijk : ndarray
        """
        AtA = self.global_gram(self.A_i)    #Internally Column reduce
        NumeratorA = self.np.zeros(self.A_i.shape).astype(self.A_i.dtype)
        DenominatorA = self.np.zeros(self.A_i.shape).astype(self.A_i.dtype)
        for x in range(self.m):
            """Compute Rx"""
            #print(self.X_ijk[x].shape,self.A_j.shape)
            XAj = self.column_mm(self.X_ijk[x], self.A_j)  #Internally row reduce
            AtXA = self.row_mm(self.A_i.T, XAj)            
            RAtA = self.matrix_mul(self.R_ijk[x],AtA)
            DenominatorR = self.matrix_mul(AtA,RAtA) + self.eps
            temp = self.element_op(AtXA,DenominatorR,"div")
            self.R_ijk[x] = self.element_op(self.R_ijk[x],temp, "mul")

            """Compute A"""
            if self.A_update:
                XARt = self.matrix_mul(XAj,self.R_ijk[x].T)
                AR = self.matrix_mul(self.A_i,self.R_ijk[x])
                XtAR = self.row_mm(self.X_ijk[x].T, AR)
                XtAR = self.column_broadcast(XtAR)
                NumeratorA += XARt + XtAR
                AtAR = self.matrix_mul(AtA,self.R_ijk[x])
                ARt = self.matrix_mul(self.A_i,self.R_ijk[x].T)
                ARtAtAR = self.matrix_mul(ARt, AtAR)
                AtARt = self.matrix_mul(AtA,self.R_ijk[x].T)
                ARAtARt = self.matrix_mul(AR,AtARt)
                DenominatorA += ARtAtAR + ARAtARt + self.eps
        if self.A_update:
            tempA = self.element_op(NumeratorA,DenominatorA,"div")
            self.A_i = self.element_op(self.A_i,tempA, "mul")
            self.A_j = self.row_broadcast(self.A_i)

