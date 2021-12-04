from pyDRESCALk.data_generator import *

import sys

import os
os.environ["OMP_NUM_THREADS"] = "1"
import pyDRESCALk.config as config
import pytest
config.init(0)
from pyDRESCALk.pyDRESCAL import *
from pyDRESCALk.dist_comm import *
from pyDRESCALk.utils import *

@pytest.mark.mpi
def test_data_generator():
    np.random.seed(100)
    args = parser()
    args.p_r = 2
    args.p_c = 2
    args.m = 3
    args.n = 12
    args.k = 2
    main_comm = MPI.COMM_WORLD
    rank = main_comm.rank
    comm = MPI_comm(main_comm, args.p_r, args.p_c)
    args.comm1 = comm.comm
    args.comm = comm
    args.col_comm = comm.cart_1d_column()
    args.row_comm = comm.cart_1d_row()
    args.rank = rank
    args.pgrid = [args.p_r, args.p_c]
    args.shape = [args.m, args.n]
    args.fpath = '../data/tmp/'
    dgen = data_generator(args)
    A_gen, R_gen, X_gen = dgen.fit()
    A_row = A_gen[0]
    A_col  =A_gen[1]
    if rank==0 or rank==3:
       assert np.allclose(A_row, A_col.T)
    A_col_broadcast = args.row_comm.bcast(A_col, root=args.col_comm.Get_rank())
    A_row_broadcast = args.col_comm.bcast(A_row, root=args.row_comm.Get_rank())
    assert np.allclose(A_col, A_col_broadcast)
    assert np.allclose(A_row, A_row_broadcast)


test_data_generator()



