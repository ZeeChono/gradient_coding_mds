#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 [-p port] [-i private_key] local_path remote_user remote_host remote_directory"
    echo "       local_path can be a file or a directory."
    exit 1
}

# Initialize variables
PORT=22  # Default port
PRIVATE_KEY=""

# Parse command line options
while getopts ":p:i:" opt; do
  case ${opt} in
    p )
      PORT=$OPTARG
      ;;
    i )
      PRIVATE_KEY=$OPTARG
      ;;
    \? )
      usage
      ;;
  esac
done
shift $((OPTIND -1))

# Check if the correct number of arguments is provided
if [ "$#" -lt 4 ]; then
    usage
fi

LOCAL_PATH=$1
REMOTE_USER=$2
REMOTE_HOST=$3
REMOTE_DIRECTORY=$4

# Construct the SCP command
SCP_COMMAND="scp -P $PORT"
if [ -n "$PRIVATE_KEY" ]; then
    SCP_COMMAND="$SCP_COMMAND -i $PRIVATE_KEY"
fi

# Check if the local path is a directory and add the -r option if it is
if [ -d "$LOCAL_PATH" ]; then
    SCP_COMMAND="$SCP_COMMAND -r"
fi

SCP_COMMAND="$SCP_COMMAND $LOCAL_PATH ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIRECTORY}"

# Execute the SCP command
echo "Executing: $SCP_COMMAND"
$SCP_COMMAND

# Check if the SCP command was successful
if [ $? -eq 0 ]; then
    echo "File transfer successful."
else
    echo "File transfer failed."
fi

