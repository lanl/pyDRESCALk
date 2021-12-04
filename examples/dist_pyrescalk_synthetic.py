#To run this code please run `mpirun -n 4 python dist_pyrescalk_dnations.py` in command line.




import sys


import pyDRESCALk.config as config

config.init(0)
from pyDRESCALk.pyDRESCALk import *
from pyDRESCALk.utils import *
from pyDRESCALk.dist_comm import *
from pyDRESCALk.data_generator import *
from scipy.io import loadmat



def dist_rescalk_2d_synthetic():

    args = parser()
    args.p_r = 2
    args.p_c = 2
    args.m = 3
    args.n = 12
    args.k = 2
    main_comm = MPI.COMM_WORLD
    rank = main_comm.rank
    size = main_comm.size
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
    args.size = size
    args.np = np
    args.fname = 'synthetic'
    args.start_k = 1
    args.end_k = 4
    args.itr = 500
    args.init = 'rand'
    args.noise_var = 0.015
    args.verbose = True
    args.norm = 'fro'
    args.method = 'mu'
    args.precision = np.float32
    args.results_path = '../Results/'
    pyDRESCALk(X_gen ,factors=None, params=args).fit()



dist_rescalk_2d_synthetic()

