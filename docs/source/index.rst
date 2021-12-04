.. pyDRESCALk documentation master file, created by
   sphinx-quickstart on Fri Dec  3 21:44:28 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyDRESCALk's documentation!
======================================
pyDRESCALk is a software package for applying non-negative RESCAL decomposition in a distributed fashion to large datasets. It can be utilized for decomposing relational datasets. It can minimize the difference between reconstructed data and the original data through Frobenius norm.  Additionally, the Custom Clustering algorithm allows for automated determination for the number of Latent features.

Features
========================

* Ability to decompose relational datasets.
* Utilization of MPI4py for distributed operation.
* Distributed random initializations.
* Distributed Custom Clustering algorithm for estimating automated latent feature number (k) determination.
* Objective of minimization of Frobenius norm.
* Support for distributed CPUs/GPUs.
* Support for Dense/Sparse data.
* Demonstrated scaling performance upto 10TB of dense and 9Exabytes of Sparse data.


Scalability
========================
pyDRESCALk Scales from laptops to clusters. The library is convenient on a laptop. It can be installed easily  with conda or pip and extends the matrix decomposition from a single core to numerous cores across nodes.
pyDRESCALk is efficient and has been tested on powerful servers across LANL and Oakridge scaling beyond 1000+ nodes.
This library facilitates the transition between single-machine to large scale cluster so as to enable users to both start simple and scale up when necessary.


Installation
========================

.. code-block:: console

   git clone https://github.com/lanl/pyDRESCALk.git
   cd pyDRESCALk
   conda create --name pyDRESCALk python=3.7.1 openmpi mpi4py
   source activate pyDRESCALk
   python setup.py install


Usage Example
========================
We provide a sample dataset that can be used for estimation of k:

.. code-block:: python

   '''Imports block'''

   import sys
   import pyDRESCALk.config as config
   config.init(0)
   from pyDRESCALk.pyDRESCALk import *
   from pyDRESCALk.data_io import *
   from pyDRESCALk.dist_comm import *
   from scipy.io import loadmat
   from mpi4py import MPI
   comm = MPI.COMM_WORLD
   args = parse()
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

Indices and tables
========================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules


Indices and tables
========================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`





