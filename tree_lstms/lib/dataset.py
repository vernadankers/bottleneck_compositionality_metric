from .tree import CustomTree
from nltk import Tree
import torch.utils.data as data
import pytreebank
import torch
import random
import os
import math
from sklearn.model_selection import KFold


PAD = 0
UNK = 1
PAD_WORD = '<pad>'
UNK_WORD = '<unk>'


class Dataset(data.Dataset):
    """
    Astract Dataset class, with functions used by multiple datsets.
    """

    def shuffle(self):
        """
        Shuffle the order in which datapoints will be retrieved.
        """
        random.shuffle(self.order)

    def __len__(self):
        """
        Measure the number of datapoints in the dataset.
        Returns:
            - size (int): number of datapoints
        """
        return self.size

    def __getitem__(self, index, bl):
        """
        Extract one individual datapoint.
        Args:
            - index (int): its position in the dataset
            - bl (boolean): whether it's a SST baseline example or not
        Returns:
            - sentence (list of str)
            - a Tree object
            - the sentence encoded in vocab item nums (list of int)
            - the final label of this example
        """
        sentence = (self.baseline_sentences if bl else self.sentences)[index]
        idx = self.vocab.convertToIdx(sentence, UNK_WORD)
        return sentence, self.trees[index], idx, self.labels[index]

    def get_batch(self, index):
        """
        Get the batch with the number 'index'.
        Args:
            - index (int): batch number
            - baseline(bool): whether it's a baseline example
        Returns:
            - sentences (list of lists of str)
            - trees (list of Tree objects)
            - indices (torch.LongTensor)
            - labels (torch.FloatTensor)
        """
        # Index is the i-th batch, start at i * batchsize and fill batch
        index = index * self.batchsize
        batch = [self.__getitem__(self.order[i], bl=self.baseline)
                 for i in range(index, min(self.size, index + self.batchsize))]
        snts, trees, idx, labels = zip(*batch)

        # Pad the vocabulary indices matrix
        max_length = max([len(x) for x in idx])
        idx = list(idx)
        for i, x in enumerate(idx):
            idx[i] = x + (max_length - len(x)) * [0]
        return snts, trees, torch.LongTensor(idx), torch.FloatTensor(labels)

    def transform_dataset(self, seed, dataset, data_setup="regular"):
        """
        Depending on the setting 'data_setup' in args, we may want to modify
        the dataset.
        Args:
            - dataset: dict with a train / dev / test portion
            - data_setup (str): "regular" / "all" / "fold_X"
        Returns:
            - modified dataset dictionary
        """
        if data_setup == "regular":
            return dataset
        elif data_setup == "all":
            # Include all examples in the training set
            dataset["train"] = dataset["train"] + dataset["dev"] + \
                dataset["test"]
            return dataset

        # Otherwise, we're doing 4-fold cross validation
        assert "fold_" in data_setup
        full_data = dataset["train"] + dataset["dev"] + dataset["test"]
        random.Random(seed).shuffle(full_data)
        folds = list(KFold(n_splits=4, shuffle=False).split(full_data))
        fold_num = int(data_setup.split('_')[-1])
        dataset["train"] = [full_data[i] for i in folds[fold_num][0]]
        dataset["dev"] = [full_data[i] for i in folds[fold_num][1]]
        dataset["test"] = dataset["dev"]
        return dataset


class ArithmeticDataset(Dataset):
    def __init__(self, path, args, vocab, is_train=False):
        self.name = "arithmetic"
        self.baseline = args.baseline
        self.num_classes = 1
        self.vocab = vocab

        # Now read input-output from file and construct Tree objects
        src_path = os.path.join(args.data_path, path)
        tgt_path = os.path.join(args.data_path, path)

        self.trees, self.sentences = [], []
        for sent in open(src_path + '.src', encoding="utf-8").readlines():
            t, s = self.tree_from_string(sent.strip())
            self.trees.append(t)
            self.sentences.append(
                s.replace('( ', " ").replace(" )", " ").split())

        with open(tgt_path + '.tgt', 'r') as f:
            self.labels = list(map(lambda x: float(x), f.readlines()))

        # Compute basic dataset statistics
        self.num_batches = math.ceil(len(self.labels) / args.batchsize)
        self.size = len(self.labels)
        self.batchsize = args.batchsize
        self.order = list(range(len(self.labels)))

    def tree_from_string(self, string_repr):
        """
        Generate arithmetic expression from string.
        """
        if string_repr.split()[0] != "(":
            string_repr = f"( {string_repr} )"

        # Insert nonterminal nodes
        nltk_format = []
        for symbol in string_repr.split():
            nltk_format.append(symbol)
            if symbol == '(':
                nltk_format.append('NT')
        nltk_format = ' '.join(nltk_format)

        # Assign characters a number indicating order in the input
        tree = []
        idx = 0
        for w in nltk_format.split():
            if w != "(" and w != ")":
                w = str(idx)
                idx += 1
            tree.append(w)

        # Now move the operators into the position of the dummy variables
        tree = self.remove_non_terminals(
            Tree.fromstring(" ".join(tree)), nltk_format)
        return CustomTree(tree, nltk_format), nltk_format

    def remove_non_terminals(self, tree, string_repr):
        if len(str(tree).split()) == 1:
            return tree
        non_terminal = string_repr.replace(")", "").replace(
            "(", "").split()[int(tree.label())]
        assert "NT" in non_terminal, tree

        if len([c for c in tree]) == 1:
            return tree[0]
        tree.set_label(tree[1])
        tree.__delitem__(1)
        self.remove_non_terminals(tree[0], string_repr)
        self.remove_non_terminals(tree[1], string_repr)
        return tree


class SentimentDataset(Dataset):
    def tree_from_string(self, string_repr):
        string_repr = string_repr.lower().replace("(", " ( ").replace(")", " ) ")
        string_repr = ' '.join(string_repr.split())

        # Assign characters a number indicating order in the input
        tree = []
        idx = 0
        for w in string_repr.split():
            if w != "(" and w != ")":
                w = str(idx)
                idx += 1
            tree.append(w)
        nltk_str2 = " ".join(tree)
        tree = Tree.fromstring(nltk_str2)
        return CustomTree(tree, string_repr, sentiment_tree=True)

    def remove_labels(self, line):
        return line.replace("(4", " <pad> ").replace("(3", " <pad> ").replace(
            "(2", " <pad> ").replace("(1", " <pad> ").replace(
            "(0", " <pad> ").replace("(", "").replace(")", "").split()

    def extract_sentences(self, dataset):
        sentences = []
        bl_sentences = []
        for sample in dataset:
            sentence = str(sample).lower()
            bl_sentences.append(self.remove_labels(self.transform(sentence)))
            sentences.append(self.remove_labels(sentence))
        return sentences, bl_sentences

    def transform(self, input):
        input = input.split()
        for i, w in enumerate(input[:-1]):
            if "(4" in w and ")" in input[i + 1]:
                input[i + 1] = "4)"
            elif "(3" in w and ")" in input[i + 1]:
                input[i + 1] = "3)"
            elif "(2" in w and ")" in input[i + 1]:
                input[i + 1] = "2)"
            elif "(1" in w and ")" in input[i + 1]:
                input[i + 1] = "1)"
            elif "(0" in w and ")" in input[i + 1]:
                input[i + 1] = "0)"
        return " ".join(input)


class SSTDataset(SentimentDataset):
    def __init__(self, path, args, vocab, is_train=False):
        super().__init__()
        self.name = "sentiment"
        self.baseline = args.baseline
        self.vocab = vocab
        self.num_classes = 5
        self.batchsize = args.batchsize

        sst_dataset = pytreebank.load_sst()
        sst_dataset = self.transform_dataset(
            args.seed, sst_dataset, args.data_setup)
        self.labels = [int(x.label) for x in sst_dataset[path]]
        sentences, baseline_sentences = self.extract_sentences(
            sst_dataset[path])

        self.sentences = []
        self.baseline_sentences = []
        self.trees = []
        zipped = zip(sst_dataset[path], sentences, baseline_sentences)
        for k, (x, s, b) in enumerate(zipped):
            tree = self.tree_from_string(str(x))
            self.sentences.append(s)
            self.baseline_sentences.append(b)
            self.trees.append(tree)

        self.num_batches = math.ceil(len(self.labels) / self.batchsize)
        self.size = len(self.labels)
        self.order = list(range(len(self.labels)))


class SentimentDatasetMini(SentimentDataset):
    def __init__(self, inputs, vocab, baseline):
        super().__init__()
        if type(inputs) == str:
            inputs = [inputs]
        self.baseline = baseline
        self.vocab = vocab
        self.idioms = False
        self.labels = [int(x[1]) for x in inputs]
        self.baseline_sentences, _ = self.extract_sentences(
            inputs)
        self.trees = [self.tree_from_string(str(x)) for x in inputs]
        self.num_batches = 1
        self.size = len(self.labels)
        self.batchsize = 1000
        self.order = list(range(len(inputs)))
        self.name = "sentiment"
