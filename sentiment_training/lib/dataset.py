import math
import random
import torch
import pickle
import pytreebank
import torch.utils.data as data
from nltk import Tree
from copy import deepcopy


class SentimentDataset(data.Dataset):
    def __init__(self, dataset_name, batchsize, seed, args):
        super().__init__()
        dataset = pytreebank.load_sst()
        if args.dataset_name != "standard":
            dataset = self.load_from_file(
                dataset, seed,
                args.dataset_file, args.dataset_name)[dataset_name]
        else:
            dataset = dataset[dataset_name]

        dataset = [Tree.fromstring(str(sent)) for sent in dataset]
        self.sentences = [' '.join(tree.leaves())
                          for tree in dataset]
        self.labels = [int(tree.label())
                       for tree in dataset]
        self.labels = self.labels
        self.num_batches = math.ceil(len(self.labels) / batchsize)
        self.size = len(self.labels)
        self.batchsize = batchsize
        self.order = list(range(len(self.labels)))

    def __getitem__(self, index):
        sentence = deepcopy(self.sentences[index])
        label = deepcopy(self.labels[index])
        return sentence, label

    def get_batch(self, index):
        index = index * self.batchsize
        batch = []
        for i in range(index, index + self.batchsize):
            if i < self.size:
                batch.append(self.__getitem__(self.order[i]))
        sentences, labels = zip(*batch)
        labels = torch.LongTensor(labels)
        return sentences, labels

    def shuffle(self):
        random.shuffle(self.order)

    def __len__(self):
        return self.size

    def load_from_file(self, sst, seed, filename, name):
        sst = sst["train"] + sst["dev"] + sst["test"]
        subset = pickle.load(open(filename, 'rb'))[name]
        train, validation, test = subset
        sst = {
            "train": [sst[i] for i in train],
            "dev": [sst[i] for i in validation],
            "test": [sst[i] for i in test]}
        return sst
