#To run this code please run `mpirun -n 4 python dist_pyrescalk_dnations.py` in command line.




import sys
import pyDRESCALk.config as config
config.init(0)
from pyDRESCALk.pyDRESCALk import *
from pyDRESCALk.utils import *
from pyDRESCALk.dist_comm import *
from scipy.io import loadmat



def dist_rescalk_2d_dnations():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    p_r, p_c = 2, 2

    comms = MPI_comm(comm, p_r, p_c)
    comm1 = comms.comm
    rank = comm.rank
    size = comm.size
    args = parse()
    args.size, args.rank, args.comm, args.p_r, args.p_c = size, rank, comms, p_r, p_c
    args.row_comm, args.col_comm, args.comm1 = comms.cart_1d_row(), comms.cart_1d_column(), comm1
    rank = comms.rank
    args.fpath = '../data/'
    args.fname = 'dnations'
    args.ftype = 'mat'
    args.start_k = 2
    args.end_k = 5
    args.itr = 200
    args.init = 'rand'
    args.noise_var = 0.005
    args.verbose = True
    args.norm = 'fro'
    args.method = 'mu'
    args.np = np
    args.precision = np.float32
    args.key = 'R'
    A_ij = np.moveaxis(data_read(args).read().astype(args.precision),-1,0) #Always make data of dimension mxnxn.
    print('Data dimension for rank=',rank,'=',A_ij.shape)
    args.results_path = '../Results/'
    pyDRESCALk(A_ij, factors=None, params=args).fit()



dist_rescalk_2d_dnations()

