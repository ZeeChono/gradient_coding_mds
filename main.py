# This is the entry file for running the cluster simulation

from __future__ import print_function
import time
import sys
sys.path.append('./src/')
from naive import *
from coded import *
from replication import *
from avoidstragg import *
from partial_replication import *
from partial_coded import *
from bibd import *
from spg import *
import numpy as np
from mpi4py import MPI
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if len(sys.argv) != 11 and len(sys.argv) != 14:
    print("Usage: python main.py n_procs n_rows n_cols input_dir is_real dataset is_coded n_stragglers partial_straggler_partitions coded_ver [encoding_matrix_csv L lambda]")
    sys.exit(0)

n_procs, n_rows, n_cols, input_dir, is_real, dataset, is_coded, n_stragglers , partitions, coded_ver  = [x for x in sys.argv[1:11]]
n_procs, n_rows, n_cols, is_real, is_coded, n_stragglers , partitions, coded_ver = int(n_procs), int(n_rows), int(n_cols), int(is_real), int(is_coded), int(n_stragglers), int(partitions), int(coded_ver)
input_dir = input_dir+"/" if not input_dir[-1] == "/" else input_dir


# ---- Modifiable parameters
num_itrs = 100 # Number of iterations

alpha = 1.0/n_rows #sometimes we used 0.0001 # --- coefficient of l2 regularization

# learning rate setup
learning_rate_schedule = 10.0 * np.ones(num_itrs)   # learning rate is 10.0 for each iter
# eta0=10.0
# t0 = 90.0
# learning_rate_schedule = [eta0*t0/(i + t0) for i in range(1,num_itrs+1)]


# -------------------------------------------------------------------------------------------------------------------------------

params = []
params.append(num_itrs)                 # params[0] = num of iters
params.append(alpha)                    # params[1] = alpha (l2 regularization)
params.append(learning_rate_schedule)   # params[2] = learning rate per iter


home = os.path.expanduser("~")  # home path

# number of processors
if not size == n_procs:
    print("Number of processers doesn't match!")
    sys.exit(0)
####################################################

# real data or artificial data
if not is_real:
    dataset = "artificial-data/" + str(n_rows) + "x" + str(n_cols)

# if is coded implementation
if is_coded:
    if partitions:  # partial stragglers
        if(coded_ver == 1):
            partial_replication_logistic_regression(n_procs, n_rows, n_cols, input_dir + dataset +"/partial/" + str((partitions-n_stragglers)*(n_procs-1)) + "/", n_stragglers, partitions, is_real, params)
        elif(coded_ver == 0):
            partial_coded_logistic_regression(n_procs, n_rows, n_cols, input_dir + dataset +"/partial/" + str((partitions-n_stragglers)*(n_procs-1)) + "/", n_stragglers, partitions, is_real, params)
            
    else:           # total stragglers
        if(coded_ver == 0): # Cyclcic
            coded_logistic_regression(n_procs, n_rows, n_cols, os.path.join(home, input_dir, dataset, str(n_procs-1)), n_stragglers, is_real, params)
            
        elif(coded_ver == 1):   # Repitition
            replication_logistic_regression(n_procs, n_rows, n_cols, os.path.join(home, input_dir, dataset, str(n_procs-1)), n_stragglers, is_real, params)

        elif(coded_ver ==2):    # Ignore
            avoidstragg_logistic_regression(n_procs, n_rows, n_cols, os.path.join(home, input_dir, dataset, str(n_procs-1)), n_stragglers, is_real, params)
        
        elif(coded_ver ==3):    # bibd
            B = np.array([
                [1, 1, 1, 0, 0, 0, 0],
                [1, 0, 0, 1, 1, 0, 0],
                [1, 0, 0, 0, 0, 1, 1],
                [0, 1, 0, 1, 0, 1, 0],
                [0, 1, 0, 0, 1, 0, 1],
                [0, 0, 1, 1, 0, 0, 1],
                [0, 0, 1, 0, 1, 1, 0]
            ])          #v=b=7, k=3, r=3
            params.append(B)
            params.append(3)    # L
            params.append(1)    # lamda
            bibd_logistic_regression(n_procs, n_rows, n_cols, os.path.join(home, input_dir, dataset, str(n_procs-1)), n_stragglers, is_real, params)

        elif coded_ver == 4:    # SPG
            encoding_matrix_csv = sys.argv[11]
            L = float(sys.argv[12])
            lambda_ = float(sys.argv[13])
            encoding_matrix_path = os.path.join(home, input_dir, encoding_matrix_csv)
            
            B = np.loadtxt(encoding_matrix_path, delimiter=',')  # Load encoding matrix from CSV file
            params.append(B)
            params.append(L)  # L
            params.append(lambda_)  # lambda
            spg_logistic_regression(n_procs, n_rows, n_cols, os.path.join(home, input_dir, dataset, str(n_procs-1)), n_stragglers, is_real, params)
            
else:   # not coded implementation == Naive
    # general_logistic_regression(n_procs, n_rows, n_cols, os.path.join(home, input_dir, dataset, str(n_procs-1)), n_stragglers, is_real, params)
    naive_logistic_regression(n_procs, n_rows, n_cols, os.path.join(home, input_dir, dataset, str(n_procs-1)), n_stragglers, is_real, params)

comm.Barrier()  # Barrier synchronization
MPI.Finalize()  # Terminate the MPI execution environment
