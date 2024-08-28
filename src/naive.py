from __future__ import print_function
import sys
import random
from util import *
import os
import numpy as np
import scipy.sparse as sps
import time
from mpi4py import MPI
from sklearn.metrics import roc_curve, auc
import pdb
from datetime import datetime

def naive_logistic_regression(n_procs, n_samples, n_features, input_dir, n_stragglers, is_real_data, params):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()


    num_itrs = params[0]            # num of iters
    beta=np.zeros(n_features)       # initialize the model (start from 0)
    
    # Load necessary data on master, and reserve space for metrics
    if rank == 0:
        # Load all training data
        if not is_real_data:
            X_train = load_data(input_dir+"1.dat")
            print(">> Loaded 1")
            for j in range(2,n_procs-1):
                X_temp = load_data(input_dir+str(j)+".dat")
                X_train = np.vstack((X_train, X_temp))
                print(">> Loaded "+str(j))
        else:
            X_train = load_sparse_csr(os.path.join(input_dir, "1"))     # load ~/dataset/amazon-dataset/2/1.npz
            for j in range(2,n_procs-1):
                X_temp = load_sparse_csr(os.path.join(input_dir, str(j)))   # load other npz
                X_train = sps.vstack((X_train, X_temp))                     # then stack them vertically
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
        acc = np.zeros(num_itrs)


    ##########################################################################
    ## FIRST STEP: SETUP ALL REQUIRED PARAMS AND REQ_LIST(LISTENER) OF MPI COMMUNICATION
    ##########################################################################

    # Loading data on workers
    if (rank):  # if rank != 0

        if not is_real_data:
            X_current = load_data(os.path.join(input_dir, str(rank)+".dat"))
            y = load_data(os.path.join(input_dir, "label.dat"))
        else:
            X_current = load_sparse_csr(os.path.join(input_dir, str(rank)))     # ~/dataset/amazon-dataset/2/#rank.npz
            y = load_data(os.path.join(input_dir, "label.dat"))
        # load correct input and output as: X_current and y_current
        rows_per_worker = X_current.shape[0]
        y_current=y[(rank-1)*rows_per_worker:rank*rows_per_worker]

    # Initializing relevant variables
    if (rank):  # workers

        predy = X_current.dot(beta) # prediction y^ = beta(model) * x_input

        g = -X_current.T.dot(np.divide(y_current,np.exp(np.multiply(predy,y_current))+1))   # logistic regression
        # message buffers
        send_req = MPI.Request()
        recv_reqs = []

    else:       # master

        msgBuffers = [np.zeros(n_features) for i in range(n_procs-1)]
        g = np.zeros(n_features)
        # betaset = np.zeros((num_itrs, n_features))
        timeset = np.zeros(num_itrs)
        worker_timeset=np.zeros((num_itrs, n_procs-1))    # each iter, how long does it take each worker to compute g
        # requests list
        request_set = []        # request_set contains recv_reqs (line 74-77)
        recv_reqs = []

        cnt_completed = 0

        status = MPI.Status()

        eta0=params[2] # ----- learning rate schedule
        alpha = params[1] # --- coefficient of l2 regularization
        utemp = np.zeros(n_features) # for accelerated gradient descent
   
    # Posting all Irecv requests for master and workers
    # setup listeners for receiving information before we proceed
    if (rank):
        # workers listens to the beta from the master
        for i in range(num_itrs):
            req = comm.Irecv([beta, MPI.DOUBLE], source=0, tag=i)
            recv_reqs.append(req)

    else:
        # master listens to the message buffer
        for i in range(num_itrs):
            recv_reqs = []              # received req per round
            for j in range(1, n_procs):
                req = comm.Irecv([msgBuffers[j-1], MPI.DOUBLE], source=j, tag=i)    # request object
                recv_reqs.append(req)
            request_set.append(recv_reqs)

    ########################################################################################################
    comm.Barrier()


    ##########################################################################
    ## SECOND STEP: ASSIGN JOBS TO EACH PROCESS, ENABLE COMMUNICATION
    ##########################################################################
    if rank==0:
        # timer on the master side
        orig_start_time= time.time()
        print("---- Starting Naive Iterations ----")

    # for 
    for i in range(num_itrs):
        
        if rank==0:

            if(i%10 == 0):  # report every 10th round
                print("\t >>> At Iteration %d" %(i))

            start_time = time.time()

            for l in range(1,n_procs):  # send beta to each worker
                comm.Isend([beta,MPI.DOUBLE],dest=l,tag=i)

            g[:]=0
            cnt_completed = 0

            # sum up the partial gradients
            while cnt_completed < n_procs-1:
                req_done = MPI.Request.Waitany(request_set[i], status)  # Wait for previously initiated request_set to complete
                src = status.Get_source()
                # print(f"from source: {src}")    # src = 1 and 2 which is the same as rank value
                worker_timeset[i,src-1] = time.time()-start_time
                request_set[i].pop(req_done)    # pop that request
                
                g += msgBuffers[src-1]   # add the partial gradients
                cnt_completed += 1

            grad_multiplier = eta0[i]/n_samples     # learning rate at i-th iter / num of samples
            # ---- update step for gradient descent
            # np.subtract((1-2*alpha*eta0[i])*beta , grad_multiplier*g, out=beta)

            # ---- updates for accelerated gradient descent
            ## TODO: check what this accelerated gradient descent come from -- NAG algorithm
            theta = 2.0/(i+2.0)
            ytemp = (1-theta)*beta + theta*utemp
            betatemp = ytemp - grad_multiplier*g - (2*alpha*eta0[i])*beta       # l2 regularization
            utemp = beta + (betatemp-beta)*(1/theta)
            beta[:] = betatemp      # the same model to broadcast for the next iteration
            
            ################
            ### EASY gd ####
            # print(f"LR = {grad_multiplier}")
            # beta = comput_gd(beta, g, 5e-7)
            ################

            timeset[i] = time.time() - start_time
            # betaset[i,:] = beta     # model at the i-th iteration

            ## NEW: calculate the train_loss and test loss
            Xtrain_beta = X_train.dot(beta)
            Xtest_beta = X_test.dot(beta)
            
            training_loss[i] = calculate_loss(y_train, Xtrain_beta, n_train)
            testing_loss[i] = calculate_loss(y_test, Xtest_beta, n_test)

            sigmoid_test = 2/(1+np.exp(-Xtest_beta)) - 1
            predy_test = np.where(sigmoid_test >= 0.0, 1, -1)

            acc[i] = compute_acc(y_test, predy_test)
            # area under ROC curve
            fpr, tpr, thresholds = roc_curve(y_test, sigmoid_test, pos_label=1)
            auc_loss[i] = auc(fpr,tpr)
            

        else:
            # workers calculate the partial gradients and send it back
            recv_reqs[i].Wait()
            
            # sendTestBuf = send_req.test()
            # if not sendTestBuf[0]:
            #     send_req.Cancel()

            predy = X_current.dot(beta)

            g = X_current.T.dot(np.divide(y_current,np.exp(np.multiply(predy,y_current))+1))
            g *= -1
            send_req = comm.Isend([g, MPI.DOUBLE], dest=0, tag=i)

    #####################################################################################################
    comm.Barrier()


    ##########################################################################
    ## FINAL STEP: TRAINING/TEST LOSS COMPUTATION AND SAVE WORK
    ##########################################################################
    if rank==0:
        elapsed_time = time.time() - orig_start_time
        print ("Total Time Elapsed: %.3f" %(elapsed_time))


        for i in range(num_itrs):
            print("Iteration %d: Train Loss = %5.3f, Test Loss = %5.3f, AUC = %5.3f, ACC = %5.3f, Total time taken =%5.3f"%(i, training_loss[i], testing_loss[i], auc_loss[i], acc[i], timeset[i]))

        # plot the image AUC vs time
        output_dir = "./out/"
        cumulative_time = [sum(timeset[:i+1]) for i in range(len(timeset))]
        sim_type = "naive"
        n_workers = n_procs-1
        plot_auc_vs_time(auc_loss, cumulative_time, sim_type, output_dir, n_workers, n_stragglers)

        output_dir = os.path.join(input_dir, "results")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Get the current timestamp
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # save_vector(training_loss, os.path.join(output_dir, f"naive_acc_training_loss_{timestamp}.dat"))
        # save_vector(testing_loss, os.path.join(output_dir, f"naive_acc_testing_loss_{timestamp}.dat"))
        # save_vector(auc_loss, os.path.join(output_dir, f"naive_acc_auc_{timestamp}.dat"))
        # save_vector(timeset, os.path.join(output_dir, f"naive_acc_timeset_{timestamp}.dat"))
        # save_matrix(worker_timeset, os.path.join(output_dir, f"naive_acc_worker_timeset_{timestamp}.dat"))

        save_vector(training_loss, os.path.join(output_dir, f"naive_acc_training_loss.dat"))
        save_vector(testing_loss, os.path.join(output_dir, f"naive_acc_testing_loss.dat"))
        save_vector(auc_loss, os.path.join(output_dir, f"naive_acc_auc.dat"))
        save_vector(timeset, os.path.join(output_dir, f"naive_acc_timeset.dat"))
        save_matrix(worker_timeset, os.path.join(output_dir, f"naive_acc_worker_timeset.dat"))
        print(f">>> Done with avg iter_time: {np.average(timeset)}")

    comm.Barrier()