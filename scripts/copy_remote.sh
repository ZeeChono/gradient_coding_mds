#!/bin/bash

# Variables
REMOTE_USER="master1"  # Replace with your remote username
REMOTE_DIR="~/log_naive"  # Replace with the remote directory where the files are located
LOCAL_DIR="results/lognaive3"  # Replace with the local directory where you want to save the files

# Loop to copy files
for i in $(seq 1 100); do
    FILENAME="NAIVE_$i.txt"
    echo "Copying $FILENAME..."
    scp "${REMOTE_USER}:${REMOTE_DIR}/${FILENAME}" "${LOCAL_DIR}/"
done

echo "All files copied successfully."