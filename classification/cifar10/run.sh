#!/bin/bash
# ------------------------------------------------------------------
# [Author] Robert Alexander Limas Sierra
#          Basic scripting code for training several IA models
# ------------------
dir=`pwd`
cd $dir

for model in lenet
do
    python train.py --net $model --n_epochs 400 --resume
done