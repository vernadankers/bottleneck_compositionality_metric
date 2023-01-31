import tqdm
import sys
sys.path.append("../../Tree-LSTM")
import torch
from treelstm import SentimentDatasetMini, Vocab, TreeLSTM
from nltk import Tree
import logging
import pytreebank
import numpy as np
import nltk
from collections import defaultdict
from statistics import mode
import copy
import random
import scipy
import pickle
from nltk.corpus import wordnet as wn
import argparse
import os


def load_model(seed, fold):
    model = TreeLSTM(vocab.size(), 300, 150, embeddings="glove",
                     dataset="sentiment", train_nodes="root",
                     bottleneck="dvib", num_classes=5)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    checkpoint = torch.load(
          "../../Tree-LSTM/checkpoints/sentiment/treelstm_bottleneck_"
          + f"seed={seed}/setup={fold}/model_beta=0.0.pt", map_location="cpu")
    model.load_state_dict(checkpoint["model"])

    sst = pytreebank.load_sst()
    full_data = sst["train"] + sst["dev"] + sst["test"]
    random.Random(seed).shuffle(full_data)
    num_examples = int(len(full_data) / 4)
    one = full_data[:num_examples]
    two = full_data[num_examples:num_examples*2]
    three = full_data[num_examples*2:num_examples*3]
    four = full_data[num_examples*3:]

    if fold == "first":
        test = four
    elif fold == "second":
        test = three
    elif fold == "third":
        test = two
    elif fold == "fourth":
        test = one
    return model, set(str(x) for x in test)


def get_prediction(inputs, model):
    dataset = SentimentDatasetMini(inputs, vocab, baseline=False)
    dataset.name = "sentiment"
    sentence, tree, inputs, label, is_idiom = dataset.get_batch(0)
    if torch.cuda.is_available():
        inputs = inputs.cuda()
    with torch.no_grad():
        return torch.argmax(
            model(tree, inputs, training_mode="inference",
                  apply_softmax=True)[0], dim=-1).tolist()


def sentiment_lexicon():
    """
    Create a version of the SST dataset that has the tree, the sentence,
    the word-level sentiment labels and the POS tags.
    Create a lexicon that maps a word with its pos tag to a sentiment label.
    """
    augmented_dataset = []
    lexicon = defaultdict(list)
    for fold in ["train", "dev", "test"]:
        for tree in pytreebank.load_sst()[fold]:
            new_sent = []
            labels = []
            sentence2 = str(tree).split()
            for i, w in enumerate(sentence2):
                if w[0] not in ['(', ')']:
                    new_sent.append(w.replace(')', ''))
                    labels.append(
                        int(sentence2[i - 1].replace(')', '').replace('(', '')))
            _, pos_tags = zip(*nltk.pos_tag(new_sent, tagset='universal'))
            for x, y, z in zip(new_sent, labels, pos_tags):
                lexicon[(x.lower(), z)].append(y)
            augmented_dataset.append(
                        (str(tree), new_sent, labels, list(pos_tags)))

    lexicon_transformed = defaultdict(lambda: defaultdict(set))
    for word, tag in lexicon:
        # Use the most frequent sentiment label as the sentimetn
        lexicon_transformed[tag][mode(lexicon[(word, tag)])].add(word)
    return augmented_dataset, lexicon_transformed



def adapt_sentences(augmented_dataset, lexicon):
    adapted_sentences = defaultdict(list)
    for k in range(len(augmented_dataset)):
        if k % 100 == 0:
            logging.info(f"Progress... {k}")
        tree, tokens, labels, pos_tags = copy.deepcopy(augmented_dataset[k])
        tree = tree.lower()
        positions = [z for z in list(range(len(tokens)))
                     if pos_tags[z] in ["NOUN", "VERB", "ADV", "ADJ"]]
        for position_to_change in positions:
            for _ in range(5):
                tree, tokens, labels, pos_tags = copy.deepcopy(
                    augmented_dataset[k])

                new_tree = []
                j = 0
                for i, w in enumerate(tree.split()):
                    if w[0] not in ['(', ')']:
                        word = w.replace(')', '')
                        label = labels.pop(0)
                        pos_tag = pos_tags.pop(0)

                        if position_to_change == j:
                            tmp = copy.deepcopy(lexicon[pos_tag][label])
                            if word.lower() in tmp:
                                tmp.remove(word.lower())
                            if tmp:
                                new_word = random.choice(list(tmp))
                            else:
                                new_word = None
                            if new_word is not None:
                                w = w.replace(word, new_word)
                        j += 1
                    new_tree.append(w)
                new_tree = " ".join(new_tree)
                if tree != new_tree:
                    adapted_sentences[str(tree)].append(new_tree)
    return adapted_sentences


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=str, default="first")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    glove = pickle.load(
        open("../../data/pickled_data/embeddings.pickle", 'rb'))
    vocab = Vocab(None, data=["<blank>", "<unk>", "dummy"],
                  embeddings="../../data/pickled_data/embeddings.pickle")

    # Create a sentiment lexicon using SST
    augmented_dataset, lexicon = sentiment_lexicon()

    # Create sentences where every time one word is swapped
    if os.path.exists("topographic_similarity_sentences.pickle"):
        adapted_sentences = pickle.load(
            open("topographic_similarity_sentences.pickle", 'rb'))
    else:
        adapted_sentences = adapt_sentences(augmented_dataset, lexicon)
        pickle.dump(
            dict(adapted_sentences),
            open("topographic_similarity_sentences.pickle", 'wb'))

    ranking = defaultdict(list)
    models = [load_model(i, args.fold) for i in [args.seed]]
    for model, test_set in models:
        for i, (tree, _, _, _) in tqdm.tqdm(enumerate(augmented_dataset)):
            if not tree in adapted_sentences:
                continue
            sents = adapted_sentences[tree]
            if sents and tree in test_set:
                print(len(sents))
                preds = get_prediction(
                    [tree] + random.sample(sents, min(50, len(sents))), model)
                for p in preds[1:]:
                    ranking[tree].append(abs(preds[0] - p))
    pickle.dump(
        dict(ranking),
        open(f"tps/topographic_similarity_fold={args.fold}_seed={args.seed}.pickle", 'wb'))
