# @author:  Namita Kharat,Manish Bhattarai
import numpy as np
from .data_io import *
from .dist_rescal import *
#from .dist_svd import *
from .utils import *
from numpy.random import SeedSequence, default_rng
from multiprocessing import Pool


class pyDRESCAL():
    r"""
    Performs the distributed NMF decomposition of given matrix X into factors W and H

    Parameters:
        A_ij (ndarray) : Distributed Data
        factors (tuple), optional : Distributed factors W and H
        params (class): Class which comprises following attributes
        params.init (str) : NMF initialization(rand/nnsvd)
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
        params.verbose(bool) : Flag to enable/disable display results
        params.save_factors(bool) : Flag to enable/disable saving computed factors"""

    @comm_timing()
    def __init__(self, X_ijk, factors=None, save_factors=False, params=None):
        self.X_ijk = X_ijk
        self.params = params
        self.np = self.params.np
        self.m_loc, self.n_loc, _ = len(self.X_ijk), self.X_ijk[0].shape[0], self.X_ijk[0].shape[1]
        self.init = self.params.init if self.params.init else 'rand'
        self.p_r, self.p_c, self.k = self.params.p_r, self.params.p_c, self.params.k  # params['m'], params['n'], params['p_r'], params['p_c'], params['k']
        self.p = self.p_r * self.p_c
        self.comm1 = self.params.comm1  # params['comm1']
        self.cart_1d_row, self.cart_1d_column, self.comm = self.params.row_comm, self.params.col_comm, self.params.comm  # params['row_comm'],params['col_comm'],params['main_comm']
        self.verbose = self.params.verbose if self.params.verbose else True
        self.perturbation =  self.params.perturbation if self.params.perturbation else 0
        self.rank = self.comm1.rank
        self.eps = self.np.finfo(X_ijk[0].dtype).eps
        self.params.eps = self.eps
        self.norm = var_init(self.params,'norm',default='fb')
        self.method = var_init(self.params,'method',default='mu')
        self.save_factors = save_factors
        self.params.itr = var_init(self.params,'itr',default=5000)
        self.seed=None
        self.itr = self.params.itr
        self.A_update = var_init(self.params,'A_update',default=True)
        self.symmetric = var_init(self.params,'symmetric',default=False)
        #self.recon_err = 0
        self.p = self.p_r * self.p_c
        if self.p_r == self.p_c:
            self.topo = '2d'
        else:
            self.topo = '1d'
            raise Exception('Current implementation only supports p_r==p_c')

        self.compute_global_dim()
        if factors is not None:
            self.A_i = factors[0].astype(self.X_ijk.dtype)
            self.A_j = self.cart_1d_row.bcast(self.A_i, root= self.cart_1d_column.Get_rank())
            self.cart_1d_row.barrier()
            self.R_ijk = factors[1].astype(self.X_ijk.dtype)
        else:
            self.init_factors()


    @comm_timing()
    def compute_global_dim(self):
        """Computes global dimensions m and n from given chunk sizes for any grid configuration"""
        self.loc_m, self.loc_n, self.loc_n = len(self.X_ijk), self.X_ijk[0].shape[0], self.X_ijk[0].shape[1]
        if self.cart_1d_row.Get_rank()%self.p_r==0:
            self.params.globaln=self.loc_n
        else:
            self.params.globaln=0
        self.params.globaln = self.comm1.allreduce(self.params.globaln)
        self.params.globalm = self.loc_m
        self.params.n = self.params.globaln
        self.params.m = self.params.globalm

    @comm_timing()
    def init_factors(self):
        """Initializes Rescal factors with rand/nnsvd method"""

        for i in range(self.p_r):
             #Different seed per perturbation and row
            if i==self.cart_1d_row.Get_rank():
                self.seed = self.perturbation * 9999 + i*1000+self.cart_1d_row.Get_rank()
                self._set_seed()
                self.A_i = self.np.random.rand(self.n_loc, self.k).astype(self.X_ijk[0].dtype)
            if i==self.cart_1d_column.Get_rank():
                self.seed = self.perturbation * 9999+i*1000+self.cart_1d_column.Get_rank()#Different seed per perturbation and column
                self._set_seed()
                self.A_j = self.np.random.rand(self.n_loc, self.k).astype(self.X_ijk[0].dtype)
        self.seed =self.perturbation * 9999 #Different seed per perturbation
        self._set_seed()
        self.R_ijk = self.np.stack([self.np.random.rand(self.k, self.k).astype(self.X_ijk[0].dtype) for _ in range(self.m_loc)], axis=0)
        if self.symmetric:
           self.R_ijk = self.R_ijk + self.R_ijk.transpose(0,2,1)


    def _set_seed(self):
        self.np.random.seed(self.seed)

    @count_memory()
    @comm_timing()
    def fit(self):
        r"""
        Calls the sub routines to perform distributed NMF decomposition with initialization for a given norm minimization and update method

        Returns
        -------
        W_i : ndarray
            Factor W of shape m/p_r * k
        H_j : ndarray
           Factor H of shape k * n/p_c
        recon_err : float
            Reconstruction error for NMF decomposition
        """
        for i in range(self.itr):
            self.A_i, self.A_j, self.R_ijk = rescal_algorithms_2D(self.X_ijk, self.A_i, self.A_j, self.R_ijk, params=self.params).update()
            if i % 10 == 0:
                self.R_ijk = self.np.maximum(self.R_ijk, self.eps)
                self.A_i = self.np.maximum(self.A_i, self.eps)
                self.A_j = self.np.maximum(self.A_j, self.eps)
            if i  == self.itr-1:
                self.relative_err()
                if self.verbose == True:
                    if self.rank == 0: print('relative error is:', self.recon_err)
                if self.save_factors:
                    data_write(self.params).save_factors([self.A_i, self.R_ijk])
        self.comm.Free()
        return self.A_i, self.A_j, self.R_ijk, self.recon_err

    @comm_timing()
    def normalize_features(self, Wall, Wall1, Hall):
        """Normalizes features Wall and Hall"""
        Wall_norm = Wall.sum(axis=0, keepdims=True) + self.eps
        Wall1_norm = Wall1.sum(axis=0, keepdims=True) + self.eps
        Wall_norm = self.comm1.allreduce(Wall_norm, op=MPI.SUM)
        Wall1_norm = self.comm1.allreduce(Wall1_norm, op=MPI.SUM)
        Wall /= Wall_norm
        Wall1 /= Wall1_norm
        Hall *= Wall_norm.T
        return Wall, Wall1, Hall

    @comm_timing()
    def relative_err(self):
        """Computes the relative error for NMF decomposition"""
        self.glob_norm_err = self.dist_norm([self.X_ijk[i] - self.A_i @ self.R_ijk[i] @ self.A_j.T for i in range(self.m_loc)],proc=self.p)
        self.glob_norm_X = self.dist_norm(self.X_ijk)
        self.recon_err = self.glob_norm_err / self.glob_norm_X

    @comm_timing()
    def dist_norm(self, X, proc=-1, norm='fro', axis=None):
        """Computes the distributed norm"""
        nm = self.np.linalg.norm(X)
        if proc != 1:
            nm = self.comm1.allreduce(nm ** 2)
        return self.np.sqrt(nm)

