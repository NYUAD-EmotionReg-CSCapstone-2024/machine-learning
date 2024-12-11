#!/bin/bash

NETID=ap7146
REMOTE=jubail.abudhabi.nyu.edu

# Assign SEEDV folder paths
SEEDV_PATH="C:\Users\adi\Documents\SEED-V\"

# Create datasets dir in /scratch/$NETID if it doesn't exist in the remote machine
ssh $NETID@$REMOTE "mkdir -p /scratch/$NETID/datasets"

# Copy SEED-V datasets to /scratch/$NETID/datasets
scp -r $SEEDV_PATH $NETID@$REMOTE:/scratch/$NETID/datasets