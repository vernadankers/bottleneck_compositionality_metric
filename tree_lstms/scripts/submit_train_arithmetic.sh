#!/bin/bash
if [ "${1}" = "bcm_pp" ]; then
    for beta in 0 1e-4 0.01 0.025 0.05 0.1 0.25 0.5 1 2.5
    do
        sbatch --array=1-10 -t 10:00:00 -N 1 -n 1 --cpus-per-task 1 \
            -o traces/arithmetic/treelstm_bottleneck=dvib_beta=${beta}_size=150_dropout=0_seed=%a.out \
            scripts/train_arithmetic.sh treelstm_bottleneck dvib $beta 150 0
    done
    for dropout in 0.1 0.25 0.5 0.65 0.75 0.85 0.9
    do
        sbatch --array=1-10 -t 10:00:00 -N 1 -n 1 --cpus-per-task 1 \
            -o traces/arithmetic/treelstm_bottleneck=dropout_beta=0_size=150_dropout=${dropout}_seed=%a.out \
            scripts/train_arithmetic.sh treelstm_bottleneck dropout 0 150 $dropout
        done
    for size in 125 100 75 50 25 10 5
    do
        sbatch --array=1-10 -t 10:00:00 -N 1 -n 1 --cpus-per-task 1 \
            -o traces/arithmetic/treelstm_bottleneck=size_beta=0_size=${size}_dropout=0_seed=%a.out \
            scripts/train_arithmetic.sh treelstm_bottleneck size 0 $size 0
    done
elif [ "${1}" = "bcm_tt" ]; then
    for beta in 0.25
    do
        sbatch -t 10:00:00 --array=1-10 -n 1 --cpus-per-task 1 \
            -o traces/arithmetic/treelstm_taught_bottleneck=dvib_beta=${beta}_size=150_dropout=0_seed=%a.out \
            scripts/train_arithmetic.sh treelstm_bottleneck_teacher dvib $beta 150 0
    done
    for dropout in 0.5
    do
        sbatch -t 10:00:00 --array=1-10 -n 1 --cpus-per-task 1 \
            -o traces/arithmetic/treelstm_taught_bottleneck=dropout_beta=0_size=150_dropout=${dropout}_seed=%a.out \
            scripts/train_arithmetic.sh treelstm_bottleneck_teacher dropout 0 150 $dropout
    done
    for size in 25
    do
        sbatch -t 10:00:00 --array=1-10 -n 1 --cpus-per-task 1 \
            -o traces/arithmetic/treelstm_taught_bottleneck=size_beta=0_size=${size}_dropout=0_seed=%a.out \
            scripts/train_arithmetic.sh treelstm_bottleneck_teacher size 0 $size 0
    done
else
    echo "Unknown mode '${1}'... exiting"
fi
