import nltk
from cca_core import get_cca_similarity
from scipy import spatial
import numpy as np
import pytreebank
import random
from collections import defaultdict
import math
import pickle
from nltk import Tree


### HELPER CODE FOR CLEANING A TREE INTO A STRING AND LABELLING ITS SENTIMENT
def label(tree):
    labels = []
    tree = nltk.Tree.fromstring(tree)
    root = int(tree._label)
    left = int(tree[0]._label)
    right = int(tree[1]._label)
    if left == right == root and root != 2:
        labels.append("continuity (non-neutral)")
    if left == right == root and root == 2:
        labels.append("continuity (neutral)")
    if left > 2 and right > 2 and root > 2:
        labels.append("positive")
    if left < 2 and right < 2 and root < 2:
        labels.append("negative")
    if left < root < right or right < root < left:
        labels.append("in_between")
    if (left <= 2 and right <= 2) and root > 2 and not (left == right == 2):
        labels.append("switch")
    if (left >= 2 and right >= 2) and root < 2 and not (left == right == 2):
        labels.append("switch")
    if left > 2 and right > 2 and root == 2:
        labels.append("neutral<->polarised")
    if left < 2 and right < 2 and root == 2:
        labels.append("neutral<->polarised")
    if root == 2 and not (left < root < right or right < root < left):
        labels.append("neutral")
    if left == 2 and right == 2 and root != 2:
        labels.append("neutral<->polarised")
    if left > 2 and right > 2 and root > left and root > right:
        labels.append("amplification")
    if left < 2 and right < 2 and root < left and root < right:
        labels.append("amplification")
    if left > 2 and right > 2 and root < left and root < right and root >= 2:
        labels.append("attenuation")
    if left < 2 and right < 2 and root > left and root > right and root <= 2:
        labels.append("attenuation")
    return labels


def clean(s):
    s = s.replace("(5", "").replace("(4", "").replace(
        "(3", "").replace("(2", "").replace("(1", "").replace("(0", "").replace(")", "")
    s = s.replace("( 5", "").replace("( 4", "").replace(
        "( 3", "").replace("( 2", "").replace("( 1", "").replace("( 0", "").replace(")", "")
    return s.split()


def process(tree):
    tree = Tree.fromstring(tree)
    subtrees = []
    replace = {")": "", "(4 ": "", "(3 ": "",
               "(2 ": "", "(1 ": "", "(0 ": ""}

    def process_recursively(subtree):
        for child in subtree:
            if type(child) == str:
                continue
            str_ = ' '.join(str(child).split())
            for x, y in replace.items():
                str_ = str_.replace(x, y)
            subtrees.append((float(child.label()), str_.split()))
            if len(child) == 1:
                continue
            process_recursively(child)

    str_ = ' '.join(str(tree).split())
    for x, y in replace.items():
        str_ = str_.replace(x, y)
    subtrees.append((float(tree.label()), str_.split()))
    if len(tree) != 1:
        process_recursively(tree)
    labels, sentences = zip(*subtrees)
    return len(sentences)


### CODE FOR STORING SMALL SUBSETS THAT ARE VERY COMPOSITIONA/NOT COMPOSTIONAL
def split_subset(bin_, ratio, side):
    indices, sims = zip(*bin_)
    ranking = np.argsort(sims)
    if side == "compositional":
        train = ranking[:math.ceil(ratio * len(ranking))]
    else:
        train = ranking[-math.ceil(ratio * len(ranking)):]
    train = [indices[i] for i in train]
    test = [i for i in indices if i not in train]
    return train, test


def generate_ranking_subset(sentences, similarities, ratio, side):
    full_train = []
    full_train_indices = []
    labels = [int(x[2:3]) for x in sentences]

    for label in [0, 1, 2, 3, 4]:
        tmp_sentences = [s for s, l in zip(sentences, labels)
                         if l == label]
        tmp_sims = [s for s, l, snt in zip(similarities, labels, sentences)
                    if l == label]
        lengths = [len(clean(s)) for s in tmp_sentences]
        edges = np.histogram_bin_edges(lengths)
        bins = defaultdict(list)
        binned = np.digitize(lengths, bins=edges)
        for i in range(len(binned)):
            bins[binned[i]].append((i, tmp_sims[i]))

        for bin_ in bins:
            if len(bins[bin_]) <= 1:
                continue
            train, test = split_subset(bins[bin_], ratio, side=side)
            full_train.extend([tmp_sentences[k] for k in train])
            full_train_indices.extend([tmp_sims[k] for k in train])
    return full_train, np.mean(full_train_indices)


def save_subsets(similarities, sentences, setup, taught):
    rankings = {}
    sst = pytreebank.load_sst()
    train_length = len(sst["train"])
    val = [str(x).lower() for x in sst["dev"]]
    test = [str(x).lower() for x in sst["test"]]
    _, val, test = to_indices(train=[], val=val, test=test)

    for ratio in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
        for side in ["compositional", "non-compositional"]:
            random.seed(1)
            np.random.seed(1)
            train, stat = generate_ranking_subset(
                sentences[:train_length], similarities[:train_length], ratio=ratio, side=side)
            print(f"--> Generated {side} subset with {len(train)} examples "
                  + f"and a mean TRE distance of {stat:.2f}")
            train = to_indices(train)[0]
            rankings[f"side={side}_ratio={ratio}"] = (train, val, test)
            pickle.dump(rankings, open(
                f"subsets/subsets_metric={'bcm-pp' if not taught else 'bcm-tt'}_bottleneck={setup}.pickle", "wb"))


### CODE FOR GENERATING HARD TRAIN - TEST SPLITS
def generate_hard_split(sentences, similarities, setup, taught):
    random.seed(10)
    np.random.seed(10)
    rankings = {}
    for split in ["compositional", "non-compositional", "random"]:
        train, validation, test, stats = generate_ranking_hard(
            sentences, similarities,
            ratio1=0.7, ratio2=0.8, split=split)
        length = np.mean([len(clean(x)) for x in train])
        train, validation, test = to_indices(train, validation, test)
        print(f"--> Generated {split} split with {len(train)}/{len(validation)}/{len(test)} "
              + f"examples in the train/val/test splits. "
              + f"Trainig examples have an avg. length of {length:.2f} {stats}")
        rankings[split] = (train, validation, test)
    pickle.dump(
        rankings,
        open(f"hard_splits/hard_split_metric={'bcm-pp' if not taught else 'bcm-tt'}_bottleneck={setup}.pickle", 'wb'))


def to_indices(train=[], val=[], test=[]):
    sst = pytreebank.load_sst()
    sst = {' '.join(clean(str(l).lower())): i
           for i, l in enumerate(sst["train"] + sst["dev"] + sst["test"])}
    train = [sst[' '.join(clean(s))] for s in train]
    val = [sst[' '.join(clean(s))] for s in val]
    test = [sst[' '.join(clean(s))] for s in test]
    return train, val, test


def generate_ranking_hard(similarities, sentences, ratio1, ratio2, split):
    full_train, full_validation, full_test = [], [], []
    full_train_indices, full_validation_indices, full_test_indices = [], [], []
    labels = [int(x[2:3]) for x in sentences]
    lengths_set = [len(clean(s)) for s in sentences]

    for label in [0, 1, 2, 3, 4]:
        tmp_sentences = [s for s, l in zip(sentences, labels)
                         if l == label]
        tmp_sims = [s for s, l, snt in zip(similarities, labels, sentences)
                    if l == label]

        lengths = [len(clean(s)) for s in tmp_sentences]
        edges = np.histogram_bin_edges(lengths)
        bins = defaultdict(list)
        binned = np.digitize(lengths, bins=edges)
        for i in range(len(binned)):
            bins[binned[i]].append((i, tmp_sims[i]))

        for bin_ in bins:
            if len(bins[bin_]) <= 1:
                continue
            train, validation, test = split_hard(
                bins[bin_], ratio1, ratio2, split=split)
            full_train.extend([tmp_sentences[k] for k in train])
            full_validation.extend([tmp_sentences[k] for k in validation])
            full_test.extend([tmp_sentences[k] for k in test])
            full_train_indices.extend([tmp_sims[k] for k in train])
            full_validation_indices.extend([tmp_sims[k] for k in validation])
            full_test_indices.extend([tmp_sims[k] for k in test])

    a = np.mean(full_train_indices)
    b = np.mean(full_validation_indices)
    c = np.mean(full_test_indices)
    stats = f"Avg. BCM distance for train/val/test: {a:.2f}/{b:.2f}/{c:.2f}"
    return full_train, full_validation, full_test, stats


def split_hard(bin_, ratio1, ratio2, split):
    indices, sims = zip(*bin_)
    ranking = np.argsort(sims)
    if split == "compositional":
        # Put the compositional examples at the end - in the test set
        ranking = list(reversed(ranking))
    elif split == "random":
        # Randomise order of exmaples
        random.shuffle(ranking)
    # Otherwise the noncompositional examples are at the end - in the test set
    n, m = int(ratio1 * len(ranking)), int(ratio2 * len(ranking))
    train = ranking[:n]
    train = [indices[i] for i in train]
    validation = ranking[n:m]
    validation = [indices[i] for i in validation]
    test = ranking[m:]
    test = [indices[i] for i in test]
    return train, validation, test


### CODE FOR CREATING THE RANKING
def combine_folds(weight, folds, setup, bcm_tt=False, num_dims=-1):
    """
    Average BCM distance metrics over folds and over seeds to a full dataset ranking.
    Args:
        - weight (float): beta weight, or dropout p or hidden dim
        - folds (list with strings): fold_0 | fold_1 | fold_2 | fold_3
        - setup (bottleneck to use): dvib | size | dropout
        - bcm_tt (boolean): whether to use the TRE-training setup
        - num_dims (int): number of CCA dimensions to use
    Returns:
        - list of similarities for all examples in SST dataset
    """

    def load_hidden(filename, beta, setup, bcm_tt=False):
        # Baseline model is always the same model (marked with beta=0)
        if beta == 0.0:
            filename += f"/test_beta={beta}.pickle"
        else:
            filename += f"/test_{'taught_' if bcm_tt else ''}{setup}={beta}.pickle"
        sentences, labels, test_pred, _, hidden = pickle.load(
            open(filename, 'rb'))
        return sentences, np.array(hidden), test_pred

    distances, sentences = [], []
    prefix = "../../tree_lstms/checkpoints_old/sentiment/treelstm_bottleneck_seed="
    for seed in range(1, 11):
        dists, snts = [], []

        # Iterate over folds
        for fold in folds:
            baseline = load_hidden(
                f"{prefix}{seed}/setup={fold}", .0, setup)[1]
            s, h, _ = load_hidden(
                f"{prefix}{seed}/setup={fold}", weight, setup, bcm_tt)
            dists.extend(measure_distance(
                baseline, h, apply_cca=not bcm_tt, threshold=num_dims))
            snts.extend(s)

        # Now put the order back to the original dataset's order
        full_data = list(range(len(dists)))
        random.Random(seed).shuffle(full_data)
        distances.append([dists[full_data.index(i)]
                         for i in range(len(full_data))])
        sentences.append([snts[full_data.index(i)]
                         for i in range(len(full_data))])

    # Assert that all sentences are back in *the same* order
    for s1 in sentences:
        for s2 in sentences:
            assert tuple(s1) == tuple(s2)

    # Average distances over seeds, and apply min-max normalisation
    distances = np.mean(distances, axis=0).tolist()
    mini = min(distances)
    maxi = max(distances)
    distances = [(x - mini) / (maxi - mini) for x in distances]
    return sentences[-1], distances


def min_max(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def measure_distance(hidden1, hidden2, apply_cca=False, threshold=-1):
    if apply_cca:
        cca_results = get_cca_similarity(
            hidden1.transpose(1, 0),
            hidden2.transpose(1, 0),
            epsilon=1e-7,
            compute_dirns=True)
        if threshold == -1:
            threshold = sum_threshold(cca_results["cca_coef1"], 0.9)
        hidden1 = cca_results["cca_dirns1"]
        hidden2 = cca_results["cca_dirns2"]
        sims = [spatial.distance.cosine(h1, h2)
                for h1, h2 in zip(hidden1[:threshold].transpose(1, 0),
                                  hidden2[:threshold].transpose(1, 0))]
    else:
        sims = [spatial.distance.cosine(h1, h2)
                for h1, h2 in zip(hidden1, hidden2)]
    return sims
