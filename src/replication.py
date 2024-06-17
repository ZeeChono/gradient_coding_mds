from __future__ import print_function
import sys
import random
from util import *
import os
import numpy as np
import scipy.sparse as sps
import time
from mpi4py import MPI

def replication_logistic_regression(n_procs, n_samples, n_features, input_dir, n_stragglers, is_real_data, params):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    n_workers = n_procs-1

    if (n_workers%(n_stragglers+1)):
        print("Error: n_workers must be multiple of n_stragglers+1!")
        sys.exit(0)

    ##########################################################################
    ## FIRST STEP: SETUP ALL REQUIRED PARAMS AND REQ_LIST(LISTENER) OF MPI COMMUNICATION
    ##########################################################################
    num_itrs = params[0]            # num of iters
    beta = np.zeros(n_features)     # initialize the model (start from 0)

    rows_per_worker = n_samples//(n_procs-1)

    # Assume n_workers is a multiple of s+1
    workers_per_group = n_workers // (n_stragglers+1)     # group size: #workers



    # Loading the data on workers
    if (rank):
        if not is_real_data:
            X_current = np.zeros(((1+n_stragglers)*rows_per_worker, n_features))
            y_current = np.zeros((1+n_stragglers)*rows_per_worker)
            y = load_data(input_dir+"label.dat")

            for i in range(1+n_stragglers):
                a=(rank-1)/(n_stragglers+1) # index of group
                b=(rank-1)%(n_stragglers+1) # position inside the group
                idx=(n_stragglers+1)*a+(b+i)%(n_stragglers+1)
                
                X_current[i*rows_per_worker:(i+1)*rows_per_worker,:]=load_data(input_dir+str(idx+1)+".dat")
                y_current[i*rows_per_worker:(i+1)*rows_per_worker]=y[idx*rows_per_worker:(idx+1)*rows_per_worker]

        else:   # real data
            y = load_data(os.path.join(input_dir, "label.dat"))
            # For real dataset, read in one example file and determine how many samples we have
            x_read_temp = load_sparse_csr(os.path.join(input_dir, "1")) # because all data zip are same shape
            rows_per_worker = x_read_temp.shape[0]
            y_current = np.zeros((1+n_stragglers)*rows_per_worker)

            for i in range(1+n_stragglers):     # each worker needs to load s+1 data partitions
                group_id = (rank-1) // (workers_per_group)
                group_index = (rank-1) % (workers_per_group)
                data_id = group_index*(n_stragglers+1) + i

                if i==0:
                    X_current = load_sparse_csr(os.path.join(input_dir, str(data_id+1)))
                else:
                    X_temp = load_sparse_csr(os.path.join(input_dir, str(data_id+1)))
                    X_current = sps.vstack((X_current,X_temp))
                y_current[i*rows_per_worker:(i+1)*rows_per_worker]=y[data_id*rows_per_worker:(data_id+1)*rows_per_worker]
                print(f"Rank {rank}: loop {i} with group_id={group_id}, group_index={group_index}, data_id={data_id} and X_cur has shape: {X_current.shape}")

    # Initializing relevant variables            
    if (rank):  # workers
        predy = X_current.dot(beta)
        g = -X_current.T.dot(np.divide(y_current,np.exp(np.multiply(predy,y_current))+1))
        # message buffers
        send_req = MPI.Request()
        recv_reqs = []

    else:   # master
        msgBuffers = [np.zeros(n_features) for i in range(n_procs-1)]
        g = np.zeros(n_features)
        betaset = np.zeros((num_itrs, n_features))
        timeset = np.zeros(num_itrs)
        worker_timeset=np.zeros((num_itrs, n_procs-1))  # each iter, how long does it take each worker to compute g

        request_set = []
        recv_reqs = []
        send_set = []

        cnt_groups = 0
        completed_groups = np.ndarray(workers_per_group,dtype=bool)
        completed_workers = np.ndarray(n_procs-1,dtype=bool)

        status = MPI.Status()

        eta0=params[2] # ----- learning rate
        alpha = params[1] # --- coefficient of l2 regularization
        utemp = np.zeros(n_features) # for accelerated gradient descent

    # Posting all Irecv requests for master and workers
    if (rank):
        # workers listens to the beta(model) from the master
        for i in range(num_itrs):
            req = comm.Irecv([beta, MPI.DOUBLE], source=0, tag=i)
            recv_reqs.append(req)
    else:
        # master listens to the message buffer
        for i in range(num_itrs):
            recv_reqs = []
            for j in range(1,n_procs):
                req = comm.Irecv([msgBuffers[j-1], MPI.DOUBLE], source=j, tag=i)
                recv_reqs.append(req)
            request_set.append(recv_reqs)

    ###########################################################################################
    comm.Barrier()



    ##########################################################################
    ## SECOND STEP: ASSIGN JOBS TO EACH PROCESS, ENABLE COMMUNICATION
    ##########################################################################
    if rank == 0:
        print("---- Starting Replication Iterations for " +str(n_stragglers) + " stragglers ----")
        orig_start_time = time.time()

    for i in range(num_itrs):
        if rank==0:

            if(i%10 == 0):
                print("\t >>> At Iteration %d" %(i))

            send_set[:] = []
            g[:]=0
            completed_groups[:]=False       # len = workers_per_group
            completed_workers[:]=False      # len = #workers
            cnt_groups=0
            
            start_time = time.time()
            
            for l in range(1,n_procs):
                sreq = comm.Isend([beta, MPI.DOUBLE], dest = l, tag = i)
                send_set.append(sreq)

            # as long as one group is intact, total gradient is guaranteed
            ## Note: every group is a replica of the other, if we have all the indices finished, then we good
            while cnt_groups < workers_per_group:
                req_done = MPI.Request.Waitany(request_set[i], status)
                src = status.Get_source()
                print(f"Received src: {src}")
                worker_timeset[i,src-1]=time.time()-start_time
                request_set[i].pop(req_done)

                completed_workers[src-1] = True

                # See which index in each group is this src coming from
                group_index = (src-1) % (workers_per_group)

                if (not completed_groups[group_index]): # len=workers_per_group
                    completed_groups[group_index]=True
                    g += msgBuffers[src-1]      # only take care of the gradient from unique index across all groups
                    cnt_groups += 1

            grad_multiplier = eta0[i]/n_samples
            # ---- update step for gradient descent
            # np.subtract((1-2*alpha*eta0[i])*beta , grad_multiplier*g, out=beta)

            # ---- updates for accelerated gradient descent
            theta = 2.0/(i+2.0)
            ytemp = (1-theta)*beta + theta*utemp
            betatemp = ytemp - grad_multiplier*g - (2*alpha*eta0[i])*beta
            utemp = beta + (betatemp-beta)*(1/theta)
            beta[:] = betatemp

            timeset[i] = time.time() - start_time

            betaset[i,:] = beta

            ind_set = [l for l in range(1,n_procs) if not completed_workers[l-1]]
            for l in ind_set:
                worker_timeset[i,l-1]=-1

            #MPI.Request.Waitall(send_set)
            #MPI.Request.Waitall(request_set[i])

        else:
            
            recv_reqs[i].Wait()

            sendTestBuf = send_req.test()
            if not sendTestBuf[0]:
                send_req.Cancel()
                #print("Worker " + str(rank) + " cancelled send request for Iteration " + str(i))

            predy = X_current.dot(beta)
            g = X_current.T.dot(np.divide(y_current,np.exp(np.multiply(predy,y_current))+1))
            g *= -1
            send_req = comm.Isend([g, MPI.DOUBLE], dest=0, tag=i)
            
    #############################################################################################
    comm.Barrier()



    ##########################################################################
    ## FINAL STEP: TRAINING/TEST LOSS COMPUTATION AND SAVE WORK
    ##########################################################################
    if rank==0:
        elapsed_time= time.time() - orig_start_time
        print ("Total Time Elapsed: %.3f" %(elapsed_time))
        # Load all training data
        if not is_real_data:
            X_train = load_data(input_dir+"1.dat")
            for j in range(2,n_procs-1):
                X_temp = load_data(input_dir+str(j)+".dat")
                X_train = np.vstack((X_train, X_temp))
        else:
            X_train = load_sparse_csr(os.path.join(input_dir, "1")) # load ~/dataset/amazon-dataset/2/1.npz
            for j in range(2,n_procs-1):
                X_temp = load_sparse_csr(os.path.join(input_dir, str(j)))
                X_train = sps.vstack((X_train, X_temp))
        y_train = load_data(os.path.join(input_dir, "label.dat"))
        y_train = y_train[0:X_train.shape[0]]

        # Load all testing data
        y_test = load_data(os.path.join(input_dir, "label_test.dat"))
        if not is_real_data:
            X_test = load_data(input_dir+"test_data.dat")
        else:
            X_test = load_sparse_csr(os.path.join(input_dir, "test_data"))
        # number of samples
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]

        training_loss = np.zeros(num_itrs)
        testing_loss = np.zeros(num_itrs)
        auc_loss = np.zeros(num_itrs)

        from sklearn.metrics import roc_curve, auc

        avg_time=0.0
        for i in range(num_itrs):
            beta = np.squeeze(betaset[i,:])
            predy_train = X_train.dot(beta)
            predy_test = X_test.dot(beta)
            training_loss[i] = calculate_loss(y_train, predy_train, n_train)
            testing_loss[i] = calculate_loss(y_test, predy_test, n_test)
            fpr, tpr, thresholds = roc_curve(y_test,predy_test, pos_label=1)
            auc_loss[i] = auc(fpr,tpr)
            print("Iteration %d: Train Loss = %5.3f, Test Loss = %5.3f, AUC = %5.3f, Total time taken =%5.3f"%(i, training_loss[i], testing_loss[i], auc_loss[i], timeset[i]))
            avg_time += timeset[i]

        output_dir = os.path.join(input_dir, "results")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        save_vector(training_loss, os.path.join(output_dir, "replication_acc_%d_training_loss.dat"%(n_stragglers)))
        save_vector(testing_loss, os.path.join(output_dir, "replication_acc_%d_testing_loss.dat"%(n_stragglers)))
        save_vector(auc_loss, os.path.join(output_dir, "replication_acc_%d_auc.dat"%(n_stragglers)))
        save_vector(timeset, os.path.join(output_dir, "replication_acc_%d_timeset.dat"%(n_stragglers)))
        save_matrix(worker_timeset, os.path.join(output_dir, "replication_acc_%d_worker_timeset.dat"%(n_stragglers)))
        print(f">>> Done with avg iter_time: {avg_time / num_itrs}")

    comm.Barrier()