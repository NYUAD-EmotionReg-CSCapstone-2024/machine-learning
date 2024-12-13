#!/bin/bash

NETID=ap7146
REMOTE=jubail.abudhabi.nyu.edu

# Assign SEED-V folder paths
CODE_BASE="C:/Users/adi/Desktop/Capstone/machine-learning"
SEEDV_PATH="C:/Users/adi/Documents/SEED-V"

# Define remote paths
REMOTE_ARCHIVE_DIR="/archive/$NETID/datasets"
REMOTE_HOME_DIR="/home/$NETID/machine-learning"

# Create directories on the remote server
ssh "$NETID@$REMOTE" "mkdir -p \"$REMOTE_ARCHIVE_DIR\""
ssh "$NETID@$REMOTE" "mkdir -p \"$REMOTE_HOME_DIR\""

# Copy folders to the remote scratch directory
scp -r -O "$SEEDV_PATH" "$NETID@$REMOTE:\"$REMOTE_ARCHIVE_DIR/\""
scp -r -O "$CODE_BASE" "$NETID@$REMOTE:\"$REMOTE_HOME_DIR/\""
