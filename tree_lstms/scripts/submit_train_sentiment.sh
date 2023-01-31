#!/bin/bash

# Regular dataset, BCM-post-processing metric
if [ "${1}" = "bcm_pp" ]; then
    for beta in 0 2.5e-05 0.000625 0.00125 0.0025 0.00625 0.0125 0.025
    do
        sbatch -t 10:00:00 -n 1 --cpus-per-task 1 --array=1-10 \
               -o traces/sentiment/treelstm_bottleneck_beta=${beta}_regular_seed=%a.out \
               scripts/train_sentiment.sh treelstm_bottleneck dvib $beta 150 0 "regular"
    done
    for dropout in 0.1 0.25 0.5 0.65 0.75 0.85 0.9
    do
        sbatch -t 10:00:00 -n 1 --cpus-per-task 1 --array=1-10 \
               -o traces/sentiment/treelstm_bottleneck_dropout=${dropout}_regular_seed=%a.out \
               scripts/train_sentiment.sh treelstm_bottleneck dropout 0 150 $dropout "regular"
    done
    for size in 125 100 75 50 25 10 5
    do
        sbatch -t 10:00:00 -n 1 --cpus-per-task 1 --array=1-10 \
               -o traces/sentiment/treelstm_bottleneck_size=${size}_regular_seed=%a.out \
               scripts/train_sentiment.sh treelstm_bottleneck size 0 $size 0 "regular"
    done

# Regular dataset, BCM-TRE-training metric
elif [ "${1}" = "bcm_tt" ]; then
    for beta in 2.5e-05 0.000625 0.00125 0.0025 0.00625 0.0125 0.025
    do
        sbatch -t 10:00:00 -n 1 --cpus-per-task 1 --array=1-10 \
               -o traces/sentiment/treelstm_bottleneck_teacher_beta=${beta}_regular_seed=%a.out \
               scripts/train_sentiment.sh treelstm_bottleneck_teacher dvib $beta 150 0 "regular"
    done
    for dropout in 0.1 0.25 0.5 0.65 0.75 0.85 0.9
    do
        sbatch -t 10:00:00 -n 1 --cpus-per-task 1 --array=1-10 \
               -o traces/sentiment/treelstm_bottleneck_teacher_dropout=${dropout}_regular_seed=%a.out \
               scripts/train_sentiment.sh treelstm_bottleneck_teacher dropout 0 150 $dropout "regular"
    done
    for size in 125 100 75 50 25 10 5
    do
        sbatch -t 10:00:00 -n 1 --cpus-per-task 1 --array=1-10 \
               -o traces/sentiment/treelstm_bottleneck_teacher_size=${size}_regular_seed=%a.out \
               scripts/train_sentiment.sh treelstm_bottleneck_teacher size 0 $size 0 "regular"
    done

# Regular dataset, baseline model
elif [ "${1}" == "baseline" ]; then
    sbatch -t 10:00:00 -n 1 --cpus-per-task 1 --array=1-10 \
      -o traces/sentiment/treelstm_bottleneck_baseline_seed=%a.out \
      scripts/train_sentiment.sh treelstm_bottleneck_baseline

# 4-fold training over dataset to get full ranking, with subset of settings
elif [ "${1}" == "full_ranking" ]; then
    for setup in fold_0 fold_1 fold_2 fold_3
    do
        for beta in 0 0.0025
        do
            sbatch -t 10:00:00 -n 1 --cpus-per-task 1 --mem 10000 --array=1-10 \
                   -o traces/sentiment/treelstm_bottleneck_beta=${beta}_${setup}_seed=%a.out \
                   scripts/train_sentiment.sh treelstm_bottleneck dvib $beta 150 0 $setup
        done
        for dropout in 0.65
        do
            sbatch -t 10:00:00 -n 1 --cpus-per-task 1 --mem 10000 --array=1-10 \
                   -o traces/sentiment/treelstm_bottleneck_dropout=${dropout}_${setup}_seed=%a.out \
                   scripts/train_sentiment.sh treelstm_bottleneck dropout 0 150 $dropout $setup
        done
        for size in 25
        do
            sbatch -t 10:00:00 -n 1 --cpus-per-task 1 --mem 10000 --array=1-10 \
                   -o traces/sentiment/treelstm_bottleneck_size=${size}_${setup}_seed=%a.out \
                   scripts/train_sentiment.sh treelstm_bottleneck size 0 $size 0 $setup
        done
    done
fi
