#!/bin/bash

# two variables you need to set
pdnndir=/Users/xuhe/Documents/GSoC/pdnn  # pointer to PDNN
device=gpu0  # the device to be used. set it to "cpu" if you don't have GPUs

# export environment variables
export PYTHONPATH=$PYTHONPATH:$pdnndir
export THEANO_FLAGS=mode=FAST_RUN,device=$device,floatX=float32

# prepare testing Data
echo "Prepare data for testing"

python3 PrepTestData.py wav_test

echo "Extracting test features with the DNN model ..."

python $pdnndir/cmds/run_Extract_Feats.py --data "test.pickle.gz" \
--nnet-param dnn.param --nnet-cfg dnn.cfg \
--output-file "testfeature.pickle.gz" --layer-index -1 \
--batch-size 100 >& dnn.testing.log

echo "Annotate results into Results.txt"

python3 Annotate.py
