import sys
import os

os.environ["OMP_NUM_THREADS"] = "1"
from pyDNMFk.utils import *
from mpi4py import MPI
from scipy.io import loadmat
import pytest


@pytest.mark.mpi
def test_dist_file_split():
    comms = MPI.COMM_WORLD

    A = loadmat('../data/dnations.mat')['R']
    m, n,p = A.shape
    if comms.rank == 0:
        print('Data shape is ', A.shape)
    p_r, p_c = 2, 2

    rank = comms.rank
    dtr_blk = determine_block_params(comms, (p_r, p_c), A.shape[:2])
    dtr_blk_idx = dtr_blk.determine_block_index_range_asymm()
    dtr_blk_shp = dtr_blk.determine_block_shape_asymm()
    print('rank=', rank, ' has data of idx range = ', dtr_blk_idx, ' and shape = ', dtr_blk_shp)

    if rank == 0:
        assert dtr_blk_idx == ([0, 0], [6, 6])
        assert dtr_blk_shp == [7, 7]
    elif rank == 3:
        assert dtr_blk_idx == ([7, 7], [13, 13])
        assert dtr_blk_shp == [7, 7]


def main():
    test_dist_file_split()


if __name__ == '__main__':
    main()