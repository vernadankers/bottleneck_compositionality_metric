import os
import pickle
import logging
import numpy as np
import torch

from lib import Roberta, LSTM, SentimentDataset, Trainer, set_seed, report
from config import parse_args


def load_data(args, logger):
    # get vocab object from vocab file previously written
    train_dataset = SentimentDataset("train", args.batchsize, args.seed, args)
    dev_dataset = SentimentDataset("dev", 128, args.seed, args)
    test_dataset = SentimentDataset("test", args.batchsize, args.seed, args)
    logger.info(f"Train/dev/test sizes = {len(train_dataset)}/"
                + f"{len(dev_dataset)}/{len(test_dataset)}")
    return train_dataset, dev_dataset, test_dataset


def main():
    args = parse_args()
    # global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    for k, v in vars(args).items():
        logger.info(f"{k}: {v}")

    train_dataset, dev_dataset, test_dataset = load_data(args, logger)

    if args.model_type == "LSTM":
        model = LSTM(**vars(args)).to(device)
    else:
        model = Roberta(**vars(args)).to(device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info(f"{params} trainable parameters.")

    criterion = torch.nn.NLLLoss().to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # create trainer object for training and testing
    trainer = Trainer(args, model, criterion, optimizer,
                      device, args.epochs, train_dataset, logger)
    best_checkpoint = trainer.train(
        train_dataset, dev_dataset, args.disable_tqdm)
    model.load_state_dict(best_checkpoint)

    for dataset, name in [(test_dataset, "test")]:
        fn = os.path.join(args.save, name)
        test_performance, predictions, labels = trainer.test(
            dataset, args.disable_tqdm)
        data = (dataset.sentences, labels, predictions)
        report(logger, name, test_performance)
        with open(fn + ".txt", 'w', encoding="utf-8") as f:
            for p in predictions:
                f.write(f"{p}\n")
        pickle.dump(data, open(fn + ".pickle", 'wb'))


if __name__ == "__main__":
    main()
