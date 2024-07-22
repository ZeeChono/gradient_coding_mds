# Number of times to run the script
NUM_RUNS=5

# Command to run
COMMAND="mpirun -np 3 -H localhost,w1,w2 python3 main.py 3 26215 241915 gradient_coding/dataset 1 amazon-dataset 0 1 1 0"

# Loop to run the command multiple times
for i in $(seq 1 $NUM_RUNS)
do
    echo "Run #$i"
    $COMMAND
    echo "Run #$i completed"
done