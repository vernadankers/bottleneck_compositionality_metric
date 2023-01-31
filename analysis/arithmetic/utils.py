from cca_core import get_cca_similarity, sum_threshold
from scipy import spatial
import numpy as np
import torch
import pickle


def load(model, filename):
    model_folder = f"../../tree_lstms/checkpoints/arithmetic/{model}"
    filename = filename.replace("size=150", "beta=0.0").replace(
        "dropout=0.0", "beta=0.0")
    sentences, labels, test_pred = \
        pickle.load(open(f"{model_folder}/{filename}.pickle", 'rb'))[:3]
    old_labels = [int(x) for x in open(
        "../../data/arithmetic_basic/test.tgt").readlines()]
    new_labels = [int(x) for x in open(
        "../../data/arithmetic_ambiguous/test.tgt").readlines()]
    return list(sentences), list(test_pred), old_labels, new_labels


def mse(x, y):
    return (x - y) ** 2


def min_max(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def measure_distance(hidden1, hidden2, apply_cca=False, threshold=-1):
    mse = torch.nn.MSELoss()
    if apply_cca:
        cca_results = get_cca_similarity(
            hidden1.transpose(1, 0),
            hidden2.transpose(1, 0),
            epsilon=5e-7,
            compute_dirns=True)
        if threshold == -1:
            threshold = 150
        hidden1 = cca_results["cca_dirns1"]
        hidden2 = cca_results["cca_dirns2"]
        dists = [spatial.distance.cosine(h1, h2)
                 for h1, h2 in zip(hidden1[:threshold].transpose(1, 0),
                                   hidden2[:threshold].transpose(1, 0))]
    else:
        dists = [mse(torch.FloatTensor(h1), torch.FloatTensor(h2)).item()
                 for h1, h2 in zip(hidden1, hidden2)]
    return dists


def load_pickle(seed, filename, dataset, mode, beta):
    if "teacher" in filename and (beta in [0.0, 150]):
        fn = f"../../tree_lstms/checkpoints/arithmetic/treelstm_bottleneck_seed={seed}" + \
            f"/{dataset}_{mode}={beta}.pickle"
    else:
        if "teacher" not in filename:
            fn = filename + f"/{dataset}_{mode}={beta}.pickle"
        else:
            fn = filename + f"/{dataset}_taught_{mode}={beta}.pickle"
    fn = fn.replace("dropout=0.0.pickle", "beta=0.0.pickle")
    fn = fn.replace("dropout=0.pickle", "beta=0.0.pickle")
    fn = fn.replace("size=150.pickle", "beta=0.0.pickle")

    sentences, _, _, _, hidden = pickle.load(open(fn, 'rb'))
    if type(hidden[0]) != list:
        hidden = [x.tolist() for x in hidden]
    return sentences, np.array(hidden)


def get_distances(weights, dataset, mode, apply_cca):
    mode_to_default = {"beta": 0.0, "dropout": 0, "size": 150}

    distances = dict()
    for weight in weights:
        all_dist = []
        for seed in range(1, 11):
            prefix = f"../../tree_lstms/checkpoints/arithmetic/"
            baseline = load_pickle(
                 seed, f"{prefix}treelstm_bottleneck_seed={seed}", dataset, mode, mode_to_default[mode])[-1]
            if apply_cca:
                snt, hidden_states = load_pickle(
                    seed, f"{prefix}treelstm_bottleneck_seed={seed}", dataset, mode, weight)
                dist = measure_distance(
                    baseline, hidden_states, apply_cca=True)
            else:
                snt, hidden_states = load_pickle(
                    seed, f"{prefix}treelstm_bottleneck_teacher_seed={seed}", dataset, mode, weight)
                dist = measure_distance(
                    baseline, hidden_states, apply_cca=False)

            all_dist.append(dist)
        distances[weight] = min_max(np.mean(all_dist, axis=0).tolist())
    return snt, distances
