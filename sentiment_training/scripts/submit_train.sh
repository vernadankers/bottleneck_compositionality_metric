#!/bin/bash

metric="bcm-pp"
bottleneck="size"
for ratio in 0.01 0.05 0.1 0.2 0.3 0.4 0.5
do
    for side in compositional non-compositional
    do
        sbatch --gres=gpu:1 --nodes=1 -t 2:00:00 \
            -o traces/LSTM_subset_metric=${metric}_bottleneck=${bottleneck}_side=${side}_ratio=${ratio}.out \
            scripts/train_sentiment.sh LSTM 2e-4 10 \
            "subsets/subsets_metric=${metric}_bottleneck=${bottleneck}" \
            "side=${side}_ratio=${ratio}"
        sbatch --gres=gpu:1 --nodes=1 -t 2:00:00 \
             -o traces/Roberta_subset_metric=${metric}_bottleneck=${bottleneck}_side=${side}_ratio=${ratio}.out \
             scripts/train_sentiment.sh Roberta 5e-6 5 \
             "subsets/subsets_metric=${metric}_bottleneck=${bottleneck}" \
             "side=${side}_ratio=${ratio}"
    done
done

for split in compositional non-compositional random
do
    sbatch --gres=gpu:1 --nodes=1 -t 2:00:00 \
        -o traces/LSTM_hard_split_metric=${metric}_bottleneck=${bottleneck}_split=${split}.out \
        scripts/train_sentiment.sh "LSTM" 2e-4 10 \
        "hard_splits/hard_split_metric=${metric}_bottleneck=${bottleneck}" \
        $split
    model="Roberta"
    sbatch --gres=gpu:1 --nodes=1 -t 2:00:00 \
        -o traces/Roberta_hard_split_metric=${metric}_bottleneck=${bottleneck}_split=${split}.out \
        scripts/train_sentiment.sh "Roberta" 5e-6 5 \
        "hard_splits/hard_split_metric=${metric}_bottleneck=${bottleneck}" \
        $split
done
