from __future__ import print_function
import sys
import os
import numpy as np
import random
import pandas as pd
from sklearn import  preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import sklearn.datasets as datasets
from util import *
import itertools
import math

import pdb

# This file contains code required for any preprocessing of real data, as well as splitting it into partitions 
# Currently this contains code relevant to the amazon-dataset (https://www.kaggle.com/c/amazon-employee-access-challenge)
# and dna dataset ftp://largescale.ml.tu-berlin.de/largescale/dna/

if len(sys.argv) != 7:
    print("Usage: python arrange_real_data.py n_procs input_dir real_dataset n_stragglers n_partitions partial_coded")
    sys.exit(0)

# read in the input arguments
np.random.seed(0)
n_procs, input_dir, real_dataset, n_stragglers, n_partitions, partial_coded  = [x for x in sys.argv[1:]]
n_procs, n_stragglers, n_partitions, partial_coded = int(n_procs), int(n_stragglers), int(n_partitions), int(partial_coded)

# declare cross-platform dataset directory location
home = os.path.expanduser("~")
input_dir = os.path.join(home, input_dir, real_dataset)    # input directory under the home

# load relevant data
if real_dataset=="amazon-dataset":

    print("Preparing data for "+real_dataset)
    trainData = pd.read_csv(os.path.join(input_dir, 'train.csv'))
    trainX = trainData.loc[:,'RESOURCE':].values     # input values
    trainY = trainData['ACTION'].values              # output value
    ## TODO: The sklearn document says this encoder should only be applied to the output?
    ## the loop below will convert the numbers into unique classes from 0 to ...
    relabeler = preprocessing.LabelEncoder()
    for col in range(len(trainX[0, :])):            # loop over the columns of trainX
        relabeler.fit(trainX[:, col])
        trainX[:, col] = relabeler.transform(trainX[:, col])

    # pdb.set_trace()

    trainY = 2*trainY - 1                           # remap Y, 1->1, 0->-1

    d_all_s = interactionTermsAmazon(trainX, degree=2)  # second order
    #d_all_t = interactionTermsAmazon(trainX, degree=3)  # third order
    #trainX = np.hstack((trainX, d_all_s, d_all_t))
    trainX = np.hstack((trainX, d_all_s))               # stack column-wise(2nd axis), not introducing new rows

    for col in range(len(trainX[0, :])):            # loop over again
        relabeler.fit(trainX[:, col])
        trainX[:, col] = relabeler.transform(trainX[:, col])

    trainX = np.vstack([trainX.T, np.ones(trainX.shape[0])]).T

    # split the data into train/val
    X_train, X_valid, y_train, y_valid = train_test_split(trainX, trainY, test_size=0.2, random_state=0)

    encoder = preprocessing.OneHotEncoder(sparse_output=True)
    encoder.fit(np.vstack((X_train, X_valid)))
    X_train = encoder.transform(X_train)  # Returns a sparse matrix (see numpy.sparse)
    X_valid = encoder.transform(X_valid)
    
    n_rows, n_cols = X_train.shape
    print("No. of training samples = %d, Dimension = %d"%(n_rows, n_cols))
    print("No. of testing samples = %d, Dimension = %d"%(X_valid.shape[0], X_valid.shape[1]))
    
    # Create output directory
    output_dir = input_dir
    if not partial_coded:
        output_dir = os.path.join(output_dir, str(n_procs-1))
        partitions = n_procs-1
    else:
        output_dir = os.path.join(output_dir, "partial", str((n_procs-1)*(n_partitions - n_stragglers)))
        partitions = (n_procs-1)*(n_partitions-n_stragglers)

    n_rows_per_worker = n_rows//partitions      # pack and distribute the data into #workers zips

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(1, partitions+1):
        data_matrix = X_train[(i-1)*n_rows_per_worker:i*n_rows_per_worker, :]   # divide the train_data into parts
        save_sparse_csr(os.path.join(output_dir,str(i)), data_matrix)     # save files   
        print("\t >>> Done with partition %d" % (i))

    save_vector(y_train, os.path.join(output_dir, "label.dat"))
    save_vector(y_valid, os.path.join(output_dir, "label_test.dat"))
    save_sparse_csr(os.path.join(output_dir,"test_data"), X_valid)

# dna dataset
elif real_dataset=="dna-dataset/dna":

    print("Preparing data for "+real_dataset)

    fin = open(input_dir + 'features.csv')
    trainData=  np.genfromtxt(itertools.islice(fin,0,500000,1), delimiter=',') 
    #np.genfromtxt(input_dir + 'features.csv',delimiter=',', max_rows=100000)
    trainX=trainData[:,1:]
    trainY=trainData[:,0]

    print("No. of positive labels = " + str(np.sum(trainY==1)))

    n,p = trainX.shape

    trainX=np.vstack([trainX.T,np.ones(trainX.shape[0])/math.sqrt(n)]).T

    X_train, X_valid, y_train, y_valid = train_test_split(trainX, trainY, test_size=0.2, random_state=0)

    encoder = preprocessing.OneHotEncoder(sparse=True)
    encoder.fit(np.vstack((X_train, X_valid)))
    X_train = encoder.transform(X_train)  # Returns a sparse matrix (see numpy.sparse)
    X_valid = encoder.transform(X_valid)
    
    n_rows, n_cols = X_train.shape
    print("No. of training samples = %d, Dimension = %d"%(n_rows,n_cols))
    print("No. of testing samples = %d, Dimension = %d"%(X_valid.shape[0],X_valid.shape[1]))
    
    # Create output directory
    output_dir = input_dir
    if not partial_coded:
        output_dir = output_dir + str(n_procs-1) + "/"
        partitions = n_procs-1
    else:
        output_dir = output_dir + "partial/" + str((n_procs-1)*(n_partitions - n_stragglers))+"/"
        partitions = (n_procs-1)*(n_partitions - n_stragglers)

    n_rows_per_worker = n_rows//partitions  # pack and distribute the data into #workers zips

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(1, partitions+1):
        data_matrix = X_train[(i-1)*n_rows_per_worker:i*n_rows_per_worker,:]
        save_sparse_csr(output_dir+str(i),data_matrix)        
        print("\t >>> Done with partition %d" % (i))

    save_vector(y_train, output_dir + "label.dat")
    save_vector(y_valid, output_dir + "label_test.dat")
    save_sparse_csr(output_dir + "test_data", X_valid)

    fin.close()

elif real_dataset=="covtype":
    print("Preparing data for "+real_dataset)
    trainData = datasets.fetch_covtype()
    trainX_tmp = trainData.data
    trainY_tmp = trainData.target

    desired_label_ind = np.where(trainY_tmp <= 2)[0]    # only keep label 0 and 1; two types
    trainX, dataset_bin_Y = np.take(trainX_tmp, desired_label_ind, axis=0), np.take(trainY_tmp, desired_label_ind)
    trainY = np.array([-1 if y == 1 else 1 for y in dataset_bin_Y])
    
    relabeler = preprocessing.LabelEncoder()
    for col in range(len(trainX[0, :])):
        relabeler.fit(trainX[:, col])
        trainX[:, col] = relabeler.transform(trainX[:, col])

    trainX=np.vstack([trainX.T,np.ones(trainX.shape[0])]).T     ## Adding intercept bias to linear model

    ## Stratified sampling to 50%, to avoid out of memory
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)        
    for train_index, test_index in sss.split(trainX, trainY):
        trainX, trainY = trainX[test_index], trainY[test_index]
    
    X_train, X_valid, y_train, y_valid = train_test_split(trainX, trainY, test_size=0.2, random_state=0)

    encoder = preprocessing.OneHotEncoder(sparse=True)
    encoder.fit(np.vstack((X_train, X_valid)))
    X_train = encoder.transform(X_train)  # Returns a sparse matrix (see numpy.sparse)
    X_valid = encoder.transform(X_valid)
    
    n_rows, n_cols = X_train.shape
    print("No. of training samples = %d, Dimension = %d"%(n_rows,n_cols))
    print("No. of testing samples = %d, Dimension = %d"%(X_valid.shape[0],X_valid.shape[1]))
    
    # Create output directory
    output_dir = input_dir
    if not partial_coded:
        output_dir = output_dir + str(n_procs-1) + "/"
        partitions = n_procs-1
    else:
        output_dir = output_dir + "partial/" + str((n_procs-1)*(n_partitions - n_stragglers))+"/"
        partitions = (n_procs-1)*(n_partitions - n_stragglers)

    n_rows_per_worker = n_rows//partitions

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(1, partitions+1):
        data_matrix = X_train[(i-1)*n_rows_per_worker:i*n_rows_per_worker,:]
        save_sparse_csr(os.path.join(output_dir,str(i)), data_matrix)        
        print("\t >>> Done with partition %d" % (i))

    save_vector(y_train, os.path.join(output_dir, "label.dat"))
    save_vector(y_valid, os.path.join(output_dir, "label_test.dat"))
    save_sparse_csr(os.path.join(output_dir, "test_data"), X_valid)

print("Data Setup Finished.")