#!/bin/bash
# ------------------------------------------------------------------
# [Author] Robert Alexander Limas Sierra
#          Basic scripting code for training several IA models
# ------------------
dir=`pwd`
cd $dir

for drop in $(seq 0.1 0.1 1)
do
    for model in cifar10 resnetDropout
    do
        python train_cifar10.py --net $model --n_epochs 400 --dropout $drop
    done
done