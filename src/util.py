from __future__ import print_function
import os
import numpy as np
import itertools
import scipy.special as sp
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

# ---- Data generation, saving, loading and modification routines

def load_data(input_file):
    mydata = np.loadtxt(input_file,dtype=float)
    return mydata
    
def save_sparse_csr(filename,array):    # Sparse matrices: format -> compressed sparse row (CSR)
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename+".npz")
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

def save_matrix(m, output):
    f = open(output, "w")
    for i in range(len(m)):
        print(" ".join([str(x) for x in m[i]]), file=f)
    f.close()

def save_vector(m, output):
    f = open(output, "w")
    for i in range(len(m)):
        print("%5.3f" %(m[i])+" ", file=f)
    f.close()

# generates a random matrix representing samples from a two-component GMM with identity covariance
def generate_random_matrix_normal(mu1, mu2, n_rows, n_cols):
    ctrmu2 = np.random.binomial(n_rows,0.5)
    ctrmu1 = n_rows - ctrmu2 
    mfac = 10/np.sqrt(n_cols)
    return np.concatenate((np.add(mfac*np.random.standard_normal((ctrmu1, n_cols)), mu1), np.add(mfac*np.random.standard_normal((ctrmu2, n_cols)), mu2)))

# generates a vector of random labels, each entry only has value -1 or 1
def generate_random_binvec(n):
    return np.array([np.random.randint(2)*2-1 for x in range(n)])

def interactionTermsAmazon(data, degree, hash=hash):
    new_data = []
    m,n = data.shape
    for indicies in itertools.combinations(range(n), degree):
        if not(5 in indicies and 7 in indicies) and not(2 in indicies and 3 in indicies):
            new_data.append([hash(tuple(v)) for v in data[:, indicies]])
    return np.array(new_data).T

# ---- Other routines 

def unique_rows(a):
    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    _, idx = np.unique(b, return_index=True)
    return a[idx]

# for cyclic computation matrix H and B
def getB(n_workers,n_stragglers):
    Htemp=np.random.normal(0,1,[n_stragglers,n_workers-1]) #generate a random matrix s*n
    H=np.vstack([Htemp.T,-np.sum(Htemp,axis=1)]).T #add the last column to H to be the negative sum of the previous columns, because H*1=0

    Ssets=np.zeros([n_workers,n_stragglers+1])

    for i in range(n_workers):
        Ssets[i,:]=np.arange(i,i+n_stragglers+1) #generate the data partition index for each worker
    Ssets=Ssets.astype(int)
    Ssets=Ssets%n_workers 
    B=np.zeros([n_workers,n_workers]) #B is the weighted matrix of data partition for each worker, row is the worker index, column is the data partition index
    for i in range(n_workers):
        B[i,Ssets[i,0]]=1   #set the first element of each row of B to be 1
        vtemp=-np.linalg.solve(H[:,np.array(Ssets[i,1:])],H[:,Ssets[i,0]]) #find the solution to the linear equation H[other columns]*v=H[first existing column]
        ctr=0
        for j in Ssets[i,1:]:
            B[i,j]=vtemp[ctr]   #set the rest of the elements of each row of B to be the solution
            ctr+=1

    return B

def getA(B,n_workers,n_stragglers):
    #S=np.array(list(itertools.permutations(np.hstack([np.zeros(n_stragglers),np.ones(n_workers-n_stragglers)]),n_workers)))
    #print(S)
    #S=unique_rows(S)
    
    S = np.ones((int(sp.binom(n_workers,n_stragglers)),n_workers))
    combs = itertools.combinations(range(n_workers), n_stragglers)
    i=0
    for pos in combs:
        S[i,pos] = 0
        i += 1

    (m,n)=S.shape
    A=np.zeros([m,n])
    for i in range(m):
        sp_pos=S[i,:]==1
        A[i,sp_pos]=np.linalg.lstsq(B[sp_pos,:].T,np.ones(n_workers))[0]

    return A

def compare(a,b):
    for id in range(len(a)):
        if a[id] and (not b[id]):
            return 1
        if (not a[id]) and b[id]:
            return -1
    return 0

def binary_search_row_wise(Aindex,completed,st,nd):
    if st>=nd-1:
        return st
    idx=(st+nd)/2
    cp=compare(Aindex[idx,:],completed)
    if (cp==0):
        return idx
    elif (cp==1):
        return binary_search_row_wise(Aindex,completed,st,idx)
    else:
        return binary_search_row_wise(Aindex,completed,idx+1,nd)

def calculate_indexA(boolvec):
    l = len(boolvec)
    ctr = 0
    ind = 0
    for j in range(l-1,-1, -1):
        if boolvec[j]:
            ctr = ctr+1
            ind = ind + sp.binom(l-1-j, ctr)

    return int(ind)

def calculate_loss(y,predy,n_samples):  # log loss
    return np.sum( np.log(1 + np.exp(-np.multiply(y,predy))) ) / n_samples


## Plot the auc versus timestamp and save as image
def plot_auc_vs_time(auc_loss, cumulative_time, sim_type, input_dir, n_workers, n_stragglers):
    output_images_dir = os.path.join(input_dir, "images")
    if not os.path.exists(output_images_dir):
        os.makedirs(output_images_dir)

    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_time, auc_loss, marker='o', linestyle='-', color='b')
    plt.xlabel('Cumulative Time (seconds)')
    plt.ylabel('AUC')
    plt.title('AUC vs Cumulative Time')
    plt.grid(True)

    output_file = sim_type+"_"+str(n_workers)+"_"+str(n_stragglers)
    # Save the plot as a PNG file
    plt.savefig(os.path.join(output_images_dir, output_file+".png"))

## Compute the accuracy of given result
def compute_acc(label_y, pred_y):
    return np.mean(label_y==pred_y)

## Compute normal gradient decent
def comput_gd(beta, g, alpha):
    return (beta - alpha*g)

## Compute numpy SPG encoding matrix:
def SPG_generator(N, K, L, lamda, gamma):
    # set the seed
    np.random.seed(42)
    B = np.random.binomial(1, gamma, size=(K,N))
    X = np.zeros((K,N))

    # compute the params to configure X matrix
    a = L/(K*gamma)
    b = float(L*gamma/K - L*L/(K*K)) / (gamma*gamma)
    c = (lamda/K - L*L/(K*K)) / (gamma*gamma)
    # mean vector
    u = a * np.ones(N)
    # covariance matrix
    cov = np.full((N, N), c)
    
    np.fill_diagonal(cov, b)
    if is_positive_semi_definite(cov):
        for i in range(K):
            X[i, :] = np.random.multivariate_normal(u, cov)
        SPG = X*B
    else: 
        raise RuntimeError("The covariance matrix is not PSD")

    return u, cov, SPG

## judge if the matrix is PSD
def is_positive_semi_definite(matrix):
    # Check if the matrix is symmetric
    if not np.allclose(matrix, matrix.T):
        return False
    
    # Compute the eigenvalues
    eigenvalues = np.linalg.eigvals(matrix)
    
    # Check if all eigenvalues are non-negative
    return np.all(eigenvalues >= 0)
