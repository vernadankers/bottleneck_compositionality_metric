#!/bin/bash

epochs=10
lr=2e-4
batchsize=4
seed=$SLURM_ARRAY_TASK_ID

if [ "${1}" = "treelstm_bottleneck" ]; then
        model_name="checkpoints/sentiment/treelstm_bottleneck_seed=${seed}/setup=${6}"
        mkdir $model_name
        echo $model_name
        python train.py --save $model_name --model_type treelstm_bottleneck \
            --dataset sentiment --batchsize $batchsize --lr $lr  --disable_tqdm \
            --seed $seed  --embeddings glove --epochs $epochs --dev dev \
            --train_nodes all --num_classes 5 --data_setup $6 \
            --bottleneck $2 --beta $3 --hidden_dim $4 --dropout $5 \
            --tensorboard runs/sentiment/treelstm_bottleneck=${2}_beta=${3}_size=${4}_dropout=${5}_seed=${seed}_setup=${6}
        wait
elif [ "${1}" = "treelstm_bottleneck_teacher" ]; then
        model_name="checkpoints/sentiment/treelstm_bottleneck_seed=${seed}/setup=${6}"
        mkdir $model_name
        echo $model_name
        python train.py --save $model_name --model_type treelstm_bottleneck \
            --dataset sentiment --batchsize $batchsize --lr $lr  --dev dev \
            --seed $seed  --embeddings glove --epochs $epochs  --data_setup $6  \
            --train_nodes all --num_classes 5 --data_setup $6 --disable_tqdm \
            --bottleneck $2 --beta $3 --hidden_dim $4 --dropout $5 \
            --tensorboard runs/sentiment/treelstm_taught_bottleneck=${2}_beta=${3}_size=${4}_dropout=${5}_seed=${seed}_setup=${6} \
            --teacher_model "checkpoints/sentiment/treelstm_bottleneck_seed=${seed}/setup=${6}/model_beta=0.0.pt"
        wait
elif [ "${1}" = "treelstm_bottleneck_baseline" ]; then
        model_name="checkpoints/sentiment/baseline_seed=${seed}"
        mkdir $model_name
        echo $model_name
        python train.py --save $model_name \
            --model_type treelstm_bottleneck --dataset sentiment \
      	    --batchsize $batchsize --lr $lr --seed $seed --embeddings glove \
      	    --epochs $epochs --train_nodes all --num_classes 5 \
      	    --data_setup "regular" --bottleneck dvib --beta 0  --dev dev \
      	    --hidden_dim 25 --dropout 0 --disable_tqdm \
      	    --tensorboard runs/sentiment/baseline_seed=${seed} --baseline
      	wait
elif [ "${1}" = "treelstm_bottleneck_baseline_teacher" ]; then
        model_name="checkpoints/sentiment/baseline_seed=${seed}"
        mkdir $model_name
        echo $model_name
        python train.py --save $model_name \
            --model_type treelstm_bottleneck --dataset sentiment  --dev dev \
      	    --batchsize $batchsize --lr $lr --seed $seed --embeddings glove \
      	    --epochs $epochs --train_nodes all --num_classes 5 \
      	    --data_setup "regular" --bottleneck dvib --beta 0 \
      	    --hidden_dim 25 --disable_tqdm \
      	    --tensorboard runs/sentiment/baseline_taught_seed=${seed} --baseline \
      	    --teacher_model "checkpoints/sentiment/treelstm_bottleneck_seed=${seed}/setup=regular/model_beta=0.0.pt"
      	wait
else
  echo "That model type is unknown...exiting"
fi
