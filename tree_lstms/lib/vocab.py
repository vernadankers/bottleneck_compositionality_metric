from collections import Counter
import math
import pickle


class Vocab(object):
    """
    Vocab object loosely based on harvardnlp/opennmt-py.
    """

    def __init__(self, args, filename=None, data=["<unk>"], lower=False,
                 embeddings="../data/pickled_data/embeddings.pickle"):
        self.idxToLabel = {}
        self.labelToIdx = {}
        self.frequencies = Counter()
        self.lower = lower

        # Special entries will not be pruned.
        self.specials = data

        if data is not None:
            self.addSpecials(data)
        if filename is not None:
            self.loadFile(filename)

        if args is None:
            glove = pickle.load(open(embeddings, 'rb'))
            for x in glove:
                self.add(x)
        else:
            if args.dataset != "arithmetic":
                glove = pickle.load(open(embeddings, 'rb'))
                for x in glove:
                    self.add(x)
            elif args.dataset == "sentiment" and args.baseline:
                for w in ["1", "2", "3", "4", "5", ")", "(", "dummy"]:
                    self.add(w)
            else:
                for w in open("../data/input_vocabulary.txt").readlines():
                    self.add(w.strip())

    def size(self):
        return len(self.labelToIdx)

    # Load entries from a file.
    def loadFile(self, filename):
        idx = 0
        for line in open(filename, 'r', encoding='utf8', errors='ignore'):
            token = line.rstrip('\n')
            self.add(token)
            idx += 1

    def getIndex(self, key, default=None):
        key = key.lower() if self.lower else key
        return self.labelToIdx.get(key, default)

    def getLabel(self, idx, default=None):
        try:
            return self.idxToLabel[idx]
        except KeyError:
            return default

    # Mark this `label` and `idx` as special
    def addSpecial(self, label, idx=None):
        idx = self.add(label)
        self.frequencies[idx] = math.inf

    # Mark all labels in `labels` as specials
    def addSpecials(self, labels):
        for label in labels:
            self.addSpecial(label)

    # Add `label` in the dictionary. Use `idx` as its index if given.
    def add(self, label):
        label = label.lower() if self.lower else label
        if label in self.labelToIdx:
            idx = self.labelToIdx[label]
        else:
            idx = len(self.idxToLabel)
            self.idxToLabel[idx] = label
            self.labelToIdx[label] = idx
        if label not in self.specials:
            self.frequencies[idx] += 1

    def add_all(self, sent):
        for w in sent:
            self.add(w.strip())

    # Convert `labels` to indices. Use `unkWord` if not found.
    # Optionally insert `bosWord` at the beginning and `eosWord` at the .
    def convertToIdx(self, labels, unkWord):
        vec = []
        unk = self.getIndex(unkWord)
        vec += [self.getIndex(label, default=unk) for label in labels]
        return vec

    # Convert `idx` to labels. If index `stop` is reached, convert it and return.
    def convertToLabels(self, idx, stop):
        labels = []

        for i in idx:
            labels += [self.getLabel(i)]
            if i == stop:
                break
        return labels
