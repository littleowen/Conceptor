#!/bin/bash

# two variables you need to set
pdnndir=/Users/xuhe/Documents/GSoC/pdnn  # pointer to PDNN
device=gpu0  # the device to be used. set it to "cpu" if you don't have GPUs

# export environment variables
export PYTHONPATH=$PYTHONPATH:$pdnndir
export THEANO_FLAGS=mode=FAST_RUN,device=$device,floatX=float32

# prepare Training Data
echo "Prepare data for training"

python3 PrepTrainData.py wav_train wav_valid

# train DNN model
echo "Training the DNN model ..."

python $pdnndir/cmds/run_DNN.py --train-data "train.pickle.gz" \
--valid-data "valid.pickle.gz" \
--nnet-spec "325:500:5" --wdir ./ \
--l2-reg 0.0001 --lrate "C:0.2:10" --model-save-step 5 \
--param-output-file dnn.param --cfg-output-file dnn.cfg  >& dnn.training.log


echo "Extracting Features for ELM training ..."

python $pdnndir/cmds/run_Extract_Feats.py --data "ELMtrain.pickle.gz" \
--nnet-param dnn.param --nnet-cfg dnn.cfg \
--output-file "ELMfeature.pickle.gz" --layer-index -1 \
--batch-size 100 >& ELMdnn.testing.log

echo "Training the ELM..."

python3 ELM_training.py


