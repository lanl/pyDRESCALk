import sys

import os
os.environ["OMP_NUM_THREADS"] = "1"
import pyDRESCALk.config as config
import pytest

config.init(0)
from pyDRESCALk.pyDRESCAL import *
from pyDRESCALk.dist_comm import *


@pytest.mark.mpi
def test_dist_rescal():
    np.random.seed(100)
    comm = MPI.COMM_WORLD
    m, k, n = 4, 2, 12
    A = np.random.rand(n, k)
    R = np.random.rand(m,k, k)
    X = [A @ r @ A.T for r in R]

    for grid in ([[1, 1]]):

        p_r, p_c = grid[0], grid[1]
        comms = MPI_comm(comm, p_r, p_c)
        comm1 = comms.comm
        rank = comm.rank
        size = comm.size
        args = parse()
        args.size, args.rank, args.comm1, args.comm, args.p_r, args.p_c = size, rank, comm1, comms, p_r, p_c
        args.np = np
        args.perturbation = 1
        args.m, args.n, args.k = m, n, k
        args.itr, args.init = 1000, 'rand'
        args.row_comm, args.col_comm, args.comm1 = comms.cart_1d_row(), comms.cart_1d_column(), comm1
        args.verbose = True
        for mthd in ['mu']:  # Frobenius norm, KL divergence,  and BCD implementation
            for norm in ['fro']:
                args.method, args.norm = mthd, norm
                A_i,A_j, R, rel_error = pyDRESCAL(X, factors=None, params=args).fit()
                if rank == 0: print('working on grid=', grid, 'with norm = ', norm, ' method= ', mthd, 'rel error=',
                                    rel_error)
                assert rel_error < 1e-2



#if __name__ == '__main__':
test_dist_rescal()
