import os
import copy
import pickle
import random
import logging
import numpy as np
import torch

from lib import TreeLSTM
from lib import Vocab
from lib import ArithmeticDataset, SSTDataset
from lib import Metrics
from lib import Trainer
from config import parse_args
from torch.utils.tensorboard import SummaryWriter
from lib import get_suffix


def set_seed(seed):
    """Set random seed.
    Args:
        - seed (int): seed to use, use -1 for random selection in range 0-1000
    """
    if seed == -1:
        seed = random.randint(0, 1000)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # if you are using GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_data(args, logger):
    """
    Load the datasets from file.

    Args:
        - args: the args loaded according to the setup specified in config.py
        - logger: to report stats to the user
    Returns:
        - vocab: custom Vocab object
        - train_dataset: custom ArithmeticDataset or SSTDataset object
        - dev_dataset: custom ArithmeticDataset or SSTDataset object
        - test_dataset: custom ArithmeticDataset or SSTDataset object
    """
    vocab = Vocab(args, data=["<pad>", "<unk>", "NT"])
    logger.info(f"Vocabulary size: {vocab.size()}")
    Dataset = ArithmeticDataset if args.dataset == "arithmetic" else SSTDataset

    train_dataset = Dataset(args.train, args, vocab, is_train=True)
    dev_dataset = Dataset(args.dev, args, vocab, is_train=False)
    test_dataset = Dataset(args.test, args, vocab, is_train=False)

    logger.info(f"Train / dev / test sizes: {len(train_dataset)} /"
                + f" {len(dev_dataset)} / {len(test_dataset)}")
    return vocab, train_dataset, dev_dataset, test_dataset


def report(logger, dataset, metrics, writer=None, epoch=-1, no_saving=False):
    """
    Report statistics stored in Metrics object to user.

    Args:
        - logger (logging object)
        - dataset (str): train / dev / test
        - metrics (Metrics object): containing loss, accurary etc
        - writer (tensorboard writer): write scalar values to tensorboard
        - epoch (int): epoch to report
    """
    metric_messages = []
    message = f"Epoch {epoch} " if epoch != -1 else "Testing... " + dataset

    for metric in metrics:
        metric_messages.append(f"{metric}: {metrics[metric]:.3f}")
        if writer is not None and not no_saving:
            writer.add_scalar(f"{metric}/{dataset}", metrics[metric], epoch)

    message += ' ' + "\t".join(metric_messages)
    logger.info(message)


def evaluate(dataset, save_folder, name, trainer, suffix, disable_tqdm,
             logger, no_saving):
    """
    Evaluate on the test set, and store predictions and hidden states to file.
    Args:
        - dataset (ArithmeticDataset or SSTDataset)
        - save_folder (str): folder to store checkpoints and pickled data
        - name (str): train / dev / test
        - trainer (Trainer object): we use its test function here
        - suffix (str): for further name specification in the files saved
        - disable_tqdm (bool): whether to disable TQDM tracker
        - logger (logging object): reports performance to user
    """
    # Get predictions and task performance
    test_performance, predictions, hidden, all_kl = trainer.test(
        dataset, disable_tqdm=disable_tqdm)
    report(logger, name, test_performance)

    if not no_saving:
        # Now store to file using pickle
        fn = os.path.join(save_folder, name)
        data = ([str(x) for x in dataset.trees],
                dataset.labels, predictions, all_kl, hidden)
        pickle.dump(data, open(fn + f"{suffix}.pickle", 'wb'))


def main():
    args = parse_args()

    # Init logger, device, seed and validate args
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(name)s:%(message)s")
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.debug(args)
    set_seed(args.seed)

    # Directory to store models and data
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    # Report arg values to user
    for k, v in vars(args).items():
        logger.info(f"{k}: {v}")

    # Init the dataset and the model
    dataset = load_data(args, logger)
    vocab, train_dataset, dev_dataset, test_dataset = dataset
    model = TreeLSTM(vocab.size(), **vars(args))
    if args.embeddings == "glove":
        model.init_embeddings(vocab)
    if args.pretrained_model is not None:
        checkpoint = torch.load(args.pretrained_model, map_location="cpu")
        model.load_state_dict(checkpoint["model"])

    if args.teacher_model is not None:
        tmp_args = copy.deepcopy(args)

        # Teacher may have different hidden dim, so first hidden dim
        checkpoint = torch.load(args.teacher_model, map_location="cpu")
        tmp_args.hidden_dim = checkpoint["model"]["linear1.weight"].shape[1]
        teacher = TreeLSTM(vocab.size(), **vars(tmp_args))
        teacher.load_state_dict(checkpoint["model"])

        # TODO verify performance of teacher model is up to the standard

        # Don't update teacher parameters
        for n, p in teacher.named_parameters():
            p.requires_grad = False
        # Freeze the current model's final hidden layer to be same as teacher's
        model.linear2 = teacher.linear2
        model.linear2.requires_grad = False
        teacher.eval()
    else:
        teacher = None
    model_parameters = list(
        filter(lambda p: p.requires_grad, model.parameters()))
    params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info(f"{params} trainable parameters.")

    # Create Trainer, a tensorboard and a Metrics object to store stats
    metrics = Metrics()
    num_steps = (len(train_dataset) * args.epochs) / train_dataset.batchsize
    trainer = Trainer(args, vocab, model, metrics, teacher, num_steps)
    writer = None if args.no_saving else SummaryWriter(
        log_dir=args.tensorboard)

    # Main training loop
    for epoch in range(args.epochs):
        train_loss = trainer.train(
            epoch, train_dataset, args.disable_tqdm)
        logger.info(f"Training loss: {train_loss:.2f}, beta = {trainer.beta}")
        if not args.no_validation:
            dev_performance = trainer.test(dev_dataset, args.disable_tqdm)[0]
            report(logger, "dev", dev_performance,
                   writer, epoch, args.no_saving)

    # Save the checkpoint, compute suffix for files based on args used
    suffix = get_suffix(args)
    # Only store the base models that we need as teacher models to save space
    if not args.no_saving and args.beta == 0 and args.dropout == 0 \
            and (args.hidden_dim == 150 or args.baseline):
        checkpoint = {'model': trainer.model.state_dict(), 'args': args}
        torch.save(checkpoint, f"{args.save}/model{suffix}.pt")

    # Final predictions on dev and test set
    evaluate(dev_dataset, args.save, args.dev, trainer,
             suffix, args.disable_tqdm, logger, args.no_saving)
    evaluate(test_dataset, args.save, args.test, trainer,
             suffix, args.disable_tqdm, logger, args.no_saving)


if __name__ == "__main__":
    main()
