#!/bin/bash

# Command line arguments
model_type=$1
bottleneck=$2
beta=$3
hidden_dim=$4
dropout=$5

# Standard settings determined during preliminary hyperparameter search
batchsize=32
lr=0.0002
epochs=50
seed=$SLURM_ARRAY_TASK_ID

postfix="beta=${beta}_size=${hidden_dim}_dropout=${dropout}_seed=${seed}"

# Create model folder
model_name="checkpoints/arithmetic/${1}_seed=${seed}"
mkdir $model_name
echo $model_name

if [ "${model_type}" = "treelstm_bottleneck" ]; then
    python train.py --dataset arithmetic --model_type $model_type  \
        --batchsize $batchsize --lr $lr --hidden_dim $hidden_dim --seed $seed \
        --dropout $dropout --epochs $epochs --bottleneck $bottleneck --beta $beta \
        --save $model_name --disable_tqdm \
        --tensorboard runs/arithmetic/treelstm_bottleneck=${bottleneck}_${postfix}
    wait
elif [ "${model_type}" = "treelstm_bottleneck_teacher" ]; then
    python train.py --dataset arithmetic --model_type $model_type  \
        --batchsize $batchsize --lr $lr --hidden_dim $hidden_dim --seed $seed \
        --dropout $dropout --epochs $epochs --bottleneck $bottleneck --beta $beta \
        --save $model_name --disable_tqdm \
        --tensorboard runs/arithmetic/treelstm_bottleneck=${bottleneck}_${postfix} \
        --teacher_model "checkpoints/arithmetic/treelstm_bottleneck_seed=${seed}/model_beta=0.0.pt"
    wait
else
    echo "Model type ${model_type} not available... exiting"
fi
