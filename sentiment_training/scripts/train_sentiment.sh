#!/bin/bash

for seed in 1 2 3 4 5
do
    model_name="checkpoints/sentiment_${1}_${5}_seed=${seed}_${4}"
    mkdir $model_name
    echo $model_name
    python train.py --model_type $1 --save $model_name --disable_tqdm \
        --batchsize 4 --lr $2 --seed $seed --epochs $3 \
        --dataset_name $5 --dataset_file "../analysis/sentiment/${4}.pickle"
    wait
done
