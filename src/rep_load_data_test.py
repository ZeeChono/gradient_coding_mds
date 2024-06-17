n_workers = 9
n_stragglers = 2

n_groups = n_workers // (n_stragglers+1)    # 6/3 = 2 workers per group


# s+1 groups
for rank in range(1, n_workers+1):
    for i in range(1+n_stragglers):     # each worker needs to load s+1 data partitions
        ## TODO: understand this...
        #a = (rank-1) // (n_stragglers+1) # index of group TODO: make this floor division as well
        group_id = (rank-1) // (n_groups)
        # b = (rank-1) % (n_stragglers+1) # position inside the group
        group_index = (rank-1) % (n_groups)
        # idx = (n_stragglers+1)*group_id + (group_index+i)%(n_stragglers+1)   # data indexer
        idx = group_index*(n_stragglers+1) + i
        # if i==0:
        #     X_current = load_sparse_csr(os.path.join(input_dir, str(idx+1)))
        # else:
        #     X_temp = load_sparse_csr(os.path.join(input_dir, str(idx+1)))
        #     X_current = sps.vstack((X_current,X_temp))
        
        # y_current[i*rows_per_worker:(i+1)*rows_per_worker]=y[idx*rows_per_worker:(idx+1)*rows_per_worker]
        print(f"Rank {rank}: loop {i} with a={group_id}, b={group_index}, idx={idx}")
        print(f"Load data partition: {idx+1} \n")
    print("=================================================")