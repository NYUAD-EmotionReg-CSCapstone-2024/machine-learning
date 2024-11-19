#!/bin/bash

NETID=sn3006
REMOTE=jubail.abudhabi.nyu.edu

# Assign SEEDV and DEEP folder paths
SEEDV_PATH=/data/SEED-V/
DEEP_PATH=/data/DEEP/

# Create datasets dir in /scratch/$NETID if it doesn't exist in the remote machine
ssh $NETID@$REMOTE "mkdir -p /scratch/$NETID/datasets"

# Copy SEED-V and DEEP datasets to /scratch/$NETID/datasets
scp -r $SEEDV_PATH $NETID@$REMOTE:/scratch/$NETID/datasets
scp -r $DEEP_PATH $NETID@$REMOTE:/scratch/$NETID/datasets