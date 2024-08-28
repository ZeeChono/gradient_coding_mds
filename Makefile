# No. of workers
N_PROCS=8

# Define number of workers
WORKERS := $(shell seq -s, 1 $$(($(N_PROCS) - 1)) | sed 's/[0-9]\+/w&/g')

# No. of stragglers in our coding schemes
N_STRAGGLERS=1

# For partially coded version: how many pieces of workload will one worker be handling.
N_PARTITIONS=10

# Switch to enable partial coded schemes
PARTIAL_CODED=0

# Name of the BIBD encoding file
BIBD_FILE=bibd.csv
# Name of the encoding csv
SPG_FILE=spg.csv

# L parameter for BIBD
L1=3
# L parameter for SPG
L2=2.5

# Lambda parameter for BIBD
LAMBDA1=1
# Lambda parameter for SPG
LAMBDA2=1

# Path to folder containing the data folders
DATA_FOLDER=dataset

# If using real data, enter 1
IS_REAL=1

# Dataset directory name
# eg. /home/ubuntu/dataset/amazon-dataset/...
DATASET=amazon-dataset
N_ROWS=26215		# num of input samples in trainset, ie. X1, X2, X3... Xd
N_COLS=241915		# num of features per input, ie. x1, x2, x3... xp

# DATASET=covtype
# N_ROWS=19805		# num of input samples in trainset, ie. X1, X2, X3... Xd
# N_COLS=15092		# num of features per input, ie. x1, x2, x3... xp

# Note that DATASET is automatically set to artificial-data/ (n_rows)x(n_cols)/... if IS_REAL is set to 0 \
 or artificial-data/partial/ (n_rows)x(n_cols)/... if PARTIAL_CODED is also set to 1

# Flag to let program run multiple times:
MULTIPLE = 0

# Number of iterations for the commands
NUM_TIMES ?= 100


generate_random_data:
	python3 ./src/generate_data.py $(N_PROCS) $(N_ROWS) $(N_COLS) $(DATA_FOLDER) $(N_STRAGGLERS) $(N_PARTITIONS) $(PARTIAL_CODED)

arrange_real_data:
	python3 ./src/arrange_real_data.py $(N_PROCS) $(DATA_FOLDER) $(DATASET) $(N_STRAGGLERS) $(N_PARTITIONS) $(PARTIAL_CODED)

naive:   
	mpirun -np $(N_PROCS) -H localhost,$(WORKERS) python3 main.py $(N_PROCS) $(N_ROWS) $(N_COLS) $(DATA_FOLDER) $(IS_REAL) $(DATASET) 0 $(N_STRAGGLERS) 0 0 0

cyccoded:
	mpirun -np $(N_PROCS) -H localhost,$(WORKERS) python3 main.py $(N_PROCS) $(N_ROWS) $(N_COLS) $(DATA_FOLDER) $(IS_REAL) $(DATASET) 1 $(N_STRAGGLERS) 0 0 0

repcoded:
	mpirun -np $(N_PROCS) -H localhost,$(WORKERS) python3 main.py $(N_PROCS) $(N_ROWS) $(N_COLS) $(DATA_FOLDER) $(IS_REAL) $(DATASET) 1 $(N_STRAGGLERS) 0 1 0

avoidstragg:
	mpirun -np $(N_PROCS) -H localhost,$(WORKERS) python3 main.py $(N_PROCS) $(N_ROWS) $(N_COLS) $(DATA_FOLDER) $(IS_REAL) $(DATASET) 1 $(N_STRAGGLERS) 0 2 0

# partialrepcoded:
# 	mpirun -np $(N_PROCS) python3 main.py $(N_PROCS) $(N_ROWS) $(N_COLS) $(DATA_FOLDER) $(IS_REAL) $(DATASET) 1 $(N_STRAGGLERS) $(N_PARTITIONS) 1

# partialcyccoded:
# 	mpirun -np $(N_PROCS) python3 main.py $(N_PROCS) $(N_ROWS) $(N_COLS) $(DATA_FOLDER) $(IS_REAL) $(DATASET) 1 $(N_STRAGGLERS) $(N_PARTITIONS) 0

bibd:
	mpirun -np $(N_PROCS) -H localhost,$(WORKERS) python3 main.py $(N_PROCS) $(N_ROWS) $(N_COLS) $(DATA_FOLDER) $(IS_REAL) $(DATASET) 1 $(N_STRAGGLERS) 0 3 $(BIBD_FILE) $(L1) $(LAMBDA1) 0

spg:
	mpirun -np $(N_PROCS) -H localhost,$(WORKERS) python3 main.py $(N_PROCS) $(N_ROWS) $(N_COLS) $(DATA_FOLDER) $(IS_REAL) $(DATASET) 1 $(N_STRAGGLERS) 0 4 $(SPG_FILE) $(L2) $(LAMBDA2) 0

naive_multiple_times:
	@for i in $$(seq 1 $(NUM_TIMES)); do \
	    echo "Running naive iteration $$i"; \
	    mpirun -np $(N_PROCS) -H localhost,$(WORKERS) python3 main.py $(N_PROCS) $(N_ROWS) $(N_COLS) $(DATA_FOLDER) $(IS_REAL) $(DATASET) 0 $(N_STRAGGLERS) 0 0 $$i 1 | tee ~/log_naive/NAIVE_$$i.txt; \
	done

cyccoded_multiple_times:
	@for i in $$(seq 1 $(NUM_TIMES)); do \
	    echo "Running cyccoded iteration $$i"; \
	    mpirun -np $(N_PROCS) -H localhost,$(WORKERS) python3 main.py $(N_PROCS) $(N_ROWS) $(N_COLS) $(DATA_FOLDER) $(IS_REAL) $(DATASET) 1 $(N_STRAGGLERS) 0 0 $$i 1; \
	done

avoidstragg_multiple_times:
	@for i in $$(seq 1 $(NUM_TIMES)); do \
	    echo "Running avoidstragg iteration $$i"; \
	    mpirun -np $(N_PROCS) -H localhost,$(WORKERS) python3 main.py $(N_PROCS) $(N_ROWS) $(N_COLS) $(DATA_FOLDER) $(IS_REAL) $(DATASET) 1 $(N_STRAGGLERS) 0 2 $$i 1; \
	done

bibd_multiple_times:
	@for i in $$(seq 1 $(NUM_TIMES)); do \
	    echo "Running bibd iteration $$i"; \
	    mpirun -np $(N_PROCS) -H localhost,$(WORKERS) python3 main.py $(N_PROCS) $(N_ROWS) $(N_COLS) $(DATA_FOLDER) \
		$(IS_REAL) $(DATASET) 1 $(N_STRAGGLERS) 0 3 $(BIBD_FILE) $(L1) $(LAMBDA1) $$i 1 | tee ~/log_bibd/BIBD_0_4_$$i.txt; \
	done


spg_multiple_times:
	@for i in $$(seq 1 $(NUM_TIMES)); do \
	    echo "Running spg iteration $$i"; \
	    mpirun -np $(N_PROCS) -H localhost,$(WORKERS) python3 main.py $(N_PROCS) $(N_ROWS) $(N_COLS) $(DATA_FOLDER) \
		$(IS_REAL) $(DATASET) 1 $(N_STRAGGLERS) 0 4 $(SPG_FILE) $(L2) $(LAMBDA2) $$i 1 | tee ~/log_spg/SPG_0_4_$$i.txt; \
	done


## Test Uncoded -> BIBD -> SPG
multiple_tests:
	@for i in $$(seq 1 $(NUM_TIMES)); do \
	    echo "Running iteration $$i"; \
		mpirun -np $(N_PROCS) -H localhost,$(WORKERS) python3 main.py $(N_PROCS) $(N_ROWS) $(N_COLS) $(DATA_FOLDER) $(IS_REAL) $(DATASET) 0 $(N_STRAGGLERS) 0 0 $$i 1; \
		mpirun -np $(N_PROCS) -H localhost,$(WORKERS) python3 main.py $(N_PROCS) $(N_ROWS) $(N_COLS) $(DATA_FOLDER) $(IS_REAL) $(DATASET) 1 $(N_STRAGGLERS) 0 3 $(BIBD_FILE) $(L1) $(LAMBDA1) $$i 1; \
	    mpirun -np $(N_PROCS) -H localhost,$(WORKERS) python3 main.py $(N_PROCS) $(N_ROWS) $(N_COLS) $(DATA_FOLDER) $(IS_REAL) $(DATASET) 1 $(N_STRAGGLERS) 0 4 $(SPG_FILE) $(L2) $(LAMBDA2) $$i 1; \
	done