import sys
sys.path.append("../../tree_lstms")
from statistics import mode
import pickle
import logging
from lib import SentimentDatasetMini, Vocab, TreeLSTM
import torch


def load_baseline(seed):
    model = TreeLSTM(vocab.size(), 300, 25, embeddings="glove",
                     dataset="sentiment", train_nodes="all",
                     model_type="treelstm_bottleneck",
                     num_classes=5, baseline=True)
    model.eval()
    checkpoint = torch.load(
          "../../tree_lstms/checkpoints/sentiment/baseline_"
          + f"seed={seed}/model_beta=0.0.pt", map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    return model


def get_prediction(input, model, baseline=False):
    dataset = SentimentDatasetMini(input, vocab, baseline)
    dataset.name = "sentiment"
    input = input.lower()
    sentence, tree, input, label = dataset.get_batch(0)
    if torch.cuda.is_available():
        input = input.cuda()
    p = model(tree, input, training_mode="inference")
    return p[0].argmax(dim=-1).item()


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s - %(message)s', level=logging.INFO)
    vocab = Vocab(None, data=["<pad>", "<unk>", "NT"],
                  embeddings='../../data/pickled_data/embeddings.pickle')

    for sentiment1 in [0, 1, 2, 3, 4]:
        for sentiment2 in [0, 1, 2, 3, 4]:
            prds = []
            for seed in range(1, 11):
                model = load_baseline(seed)
                prds.append(get_prediction(
                    f"(0 (0 {sentiment1}) (0 {sentiment2}))", model, True))
            print(
                f"Left = {sentiment1}, Right = {sentiment2}, Root = {mode(prds)}")
