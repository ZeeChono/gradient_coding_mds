import sys
from util import *
import os
import numpy as np
import scipy.sparse as sps
import time
from mpi4py import MPI
from sklearn.metrics import roc_curve, auc
from datetime import datetime

def spg_logistic_regression(n_procs, n_samples, n_features, dataset, input_dir, n_stragglers, is_real_data, params):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    n_workers = n_procs-1

    num_itrs = params[0]            # num of iters
    beta = np.zeros(n_features)     # initialize the model (start from 0)
    rows_per_partition = n_samples//(n_procs-1)
    B = params[3]
    L = params[4]
    lamda = params[5]

    if rank == 0:
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
        acc = np.zeros(num_itrs)

    ##########################################################################
    ## FIRST STEP: SETUP ALL REQUIRED PARAMS AND REQ_LIST(LISTENER) OF MPI COMMUNICATION
    ##########################################################################

    # Loading the data on workers
    if (rank):
        
        y = load_data(os.path.join(input_dir, "label.dat"))
        # For real dataset, read in one example file and determine how many samples we have
        x_read_temp = load_sparse_csr(os.path.join(input_dir, "1")) # because all data zip are same shape
        rows_per_partition = x_read_temp.shape[0]

        # loop through the rank-th column of encoding matrix B to allocate data to each worker
        bi = B[:, rank-1]
        first = True
        for i in range(len(bi)):
            if bi[i] != 0:     # if the i-th element non-trivial
                # print(f"Worker-{rank} load data partition {i+1}")
                if first:
                    X_current = load_sparse_csr(os.path.join(input_dir, str(i+1)))
                    y_current = y[i*rows_per_partition:(i+1)*rows_per_partition]    # y is an 1-d array
                    y_current_mod = bi[i]*y[i*rows_per_partition:(i+1)*rows_per_partition] # modifies y_current by multiplying different segments of it with corresponding elements from B matrix
                    first = False
                else:
                    X_temp = load_sparse_csr(os.path.join(input_dir, str(i+1)))
                    X_current = sps.vstack((X_current,X_temp))
                    y_current = np.hstack((y_current,y[i*rows_per_partition:(i+1)*rows_per_partition]))
                    y_current_mod = np.hstack((y_current_mod,bi[i]*y[i*rows_per_partition:(i+1)*rows_per_partition])) # modifies y_current by multiplying different segments of it with corresponding elements from B matrix
        # print(f"Worker-{rank} has load data X with shape: {X_current.shape} and y with shape {y_current.shape}")

    # Initializing relevant variables            
    if (rank):  # workers
        predy = X_current.dot(beta)
        g = -X_current.T.dot(np.divide(y_current,np.exp(np.multiply(predy,y_current))+1))
        # message buffers
        send_req = MPI.Request()
        recv_reqs = []

    else:   # master
        msgBuffers = [np.zeros(n_features) for i in range(n_procs-1)]   # partial gradietn returned by each worker
        g = np.zeros(n_features)

        A_row = np.zeros((1,n_procs-1))

        timeset = np.zeros(num_itrs)
        worker_timeset=np.zeros((num_itrs, n_procs-1))  # each iter, how long does it take each worker to compute g

        request_set = []
        recv_reqs = []
        send_set = []

        cnt_groups = 0
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
        print("---- Starting SPG Iterations for " +str(n_stragglers) + " stragglers ----")
        orig_start_time = time.time()

    for i in range(num_itrs):

        if rank==0:

            if(i%10 == 0):
                print("\t >>> At Iteration %d" %(i))

            send_set[:] = []
            g[:]=0
            completed_workers[:]=False      # len = #workers
            cnt_completed=0
            
            start_time = time.time()
            
            # Note down the sent beta
            for l in range(1,n_procs):
                sreq = comm.Isend([beta, MPI.DOUBLE], dest = l, tag = i)
                send_set.append(sreq)

            # TODO: what would be our stopping criterion?
            while cnt_completed < n_procs-1-n_stragglers:
                req_done = MPI.Request.Waitany(request_set[i], status)
                src = status.Get_source()
                # print(f"Received src: {src}")
                worker_timeset[i,src-1]=time.time()-start_time
                request_set[i].pop(req_done)

                cnt_completed += 1
                completed_workers[src-1] = True

            # get the returned index of workers
            completed_ind_set = [l for l in range(n_procs-1) if completed_workers[l]]
            
            A_row[0,completed_ind_set] = L/(L+lamda*(n_workers-n_stragglers-1)) * 1.0
            
            # docoding stage:
            g = np.squeeze(np.dot(A_row, msgBuffers)) # get the weighted sum of the gradients from workers and sqeeze it to a one dimensional array

            ################################ Amazon-dataset ################################
            if dataset == "amazon-dataset":
                grad_multiplier = eta0[i]/n_samples     # learning rate at i-th iter / num of samples
                # ---- update step for gradient descent
                # np.subtract((1-2*alpha*eta0[i])*beta , grad_multiplier*g, out=beta)

                # ---- updates for accelerated gradient descent
                theta = 2.0/(i+2.0)
                ytemp = (1-theta)*beta + theta*utemp
                betatemp = ytemp - grad_multiplier*g - (2*alpha*eta0[i])*beta       # l2 regularization
                utemp = beta + (betatemp-beta)*(1/theta)
                beta[:] = betatemp      # the same model to broadcast for the next iteration
            #################################################################################
            

            ################################ Covtype-dataset ################################
            if dataset == "covtype_bibd":
                grad_multiplier = 1e-1/n_samples    # learning rate at i-th iter / num of samples
                # grad_multiplier = eta0[i]
                # ---- update step for gradient descent
                # np.subtract((1-2*alpha*eta0[i])*beta , grad_multiplier*g, out=beta)

                # ---- updates for accelerated gradient descent                
                if i <= 100:
                    theta = 2.0/(i+2.0)
                    ytemp = (1-theta)*beta + theta*utemp
                    betatemp = ytemp - grad_multiplier*g - (2*alpha*eta0[i])*beta       # l2 regularization
                    utemp = beta + (betatemp-beta)*(1/theta)
                    beta[:] = betatemp      # the same model to broadcast for the next iteration
                beta[:] = beta - grad_multiplier*g
            #################################################################################

            timeset[i] = time.time() - start_time

            ind_set = [l for l in range(1,n_procs) if not completed_workers[l-1]]
            for l in ind_set:
                worker_timeset[i,l-1]=-1    # set not received worker time to -1
            
            ## NEW: calculate the train_loss, test loss, auc and acc
            predy_train = X_train.dot(beta)
            predy_test = X_test.dot(beta)
            training_loss[i] = calculate_loss(y_train, predy_train, n_train)
            testing_loss[i] = calculate_loss(y_test, predy_test, n_test)
            fpr, tpr, thresholds = roc_curve(y_test,predy_test, pos_label=1)
            auc_loss[i] = auc(fpr,tpr)

            sigmoid_test = 2/(1+np.exp(-predy_test)) - 1
            predy_test = np.where(sigmoid_test >= 0.0, 1, -1)
            acc[i] = compute_acc(predy_test, y_test)


        else: 
            recv_reqs[i].Wait()
            sendTestBuf = send_req.test()
            if not sendTestBuf[0]:
                send_req.Cancel()
                #print("Worker " + str(rank) + " cancelled send request for Iteration " + str(i))

            predy = X_current.dot(beta)
            g = X_current.T.dot(np.divide(y_current_mod,np.exp(np.multiply(predy,y_current))+1))
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
        

        for i in range(num_itrs):
            print("Iteration %d: Train Loss = %5.3f, Test Loss = %5.3f, AUC = %5.3f, ACC = %5.3f, Total time taken =%5.3f"%(i, training_loss[i], testing_loss[i], auc_loss[i], acc[i], timeset[i]))

        # plot the image
        cumulative_time = [sum(timeset[:i+1]) for i in range(len(timeset))]
        sim_type = "rep"
        plot_auc_vs_time(auc_loss, cumulative_time, sim_type, input_dir, n_workers, n_stragglers)

        output_dir = os.path.join(input_dir, "results")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Get the current timestamp
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # save_vector(training_loss, os.path.join(output_dir, f"spg_acc_{n_stragglers}_training_loss_{timestamp}.dat"))
        # save_vector(testing_loss, os.path.join(output_dir, f"spg_acc_{n_stragglers}_testing_loss_{timestamp}.dat"))
        # save_vector(auc_loss, os.path.join(output_dir, f"spg_acc_{n_stragglers}_auc_{timestamp}.dat"))
        # save_vector(timeset, os.path.join(output_dir, f"spg_acc_{n_stragglers}_timeset_{timestamp}.dat"))
        # save_matrix(worker_timeset, os.path.join(output_dir, f"spg_acc_{n_stragglers}_worker_timeset_{timestamp}.dat"))

        save_vector(training_loss, os.path.join(output_dir, f"spg_acc_{n_stragglers}_training_loss.dat"))
        save_vector(testing_loss, os.path.join(output_dir, f"spg_acc_{n_stragglers}_testing_loss.dat"))
        save_vector(auc_loss, os.path.join(output_dir, f"spg_acc_{n_stragglers}_auc.dat"))
        save_vector(timeset, os.path.join(output_dir, f"spg_acc_{n_stragglers}_timeset.dat"))
        save_matrix(worker_timeset, os.path.join(output_dir, f"spg_acc_{n_stragglers}_worker_timeset.dat"))
        print(f">>> Done with avg iter_time: {np.average(timeset)}")

    comm.Barrier()