import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="PyTorch (Tree)LSTM implementation.")

    # data arguments
    parser.add_argument("--dataset", choices=["sentiment", "arithmetic"],
                        default="arithmetic",
                        help="Dataset to use: arithmetic | sentiment")
    parser.add_argument("--data_path", default="../data/arithmetic_ambiguous/",
                        help="path to dataset")
    parser.add_argument("--train", default="train",
                        help="dataset name used for training")
    parser.add_argument("--dev", default="val",
                        help="dataset name used for development")
    parser.add_argument("--test", default="test",
                        help="dataset name used for testing")
    parser.add_argument("--save", default="checkpoints/",
                        help="directory to save checkpoints in")
    parser.add_argument("--vocabulary", default="../data/input_vocabulary.txt",
                        help="File w/ vocab tokens, not used for sentiment.")
    parser.add_argument("--data_setup", default="regular",
                        choices=["regular", "all", "fold_0", "fold_1",
                                 "fold_2", "fold_3"])
    parser.add_argument("--baseline", action="store_true")

    # model arguments
    parser.add_argument("--input_dim", default=150, type=int,
                        help="Size of input word vector")
    parser.add_argument("--hidden_dim", default=150, type=int,
                        help="Size of (Tree)LSTM cell state")
    parser.add_argument("--num_classes", default=1, type=int,
                        help="Number of classes in dataset")
    parser.add_argument("--model_type", default="treelstm", type=str,
                        choices=["lstm", "treelstm", "treelstm_bottleneck",
                                 "treelstm_bottleneck_teacher"],
                        help="Model to use: lstm | treelstm")
    parser.add_argument("--pretrained_model", type=str, default=None,
                        help="Name of pretrained model for initalisation.")
    parser.add_argument("--teacher_model", type=str, default=None,
                        help="Name of teacher model.")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Amount of dropout used in the (Tree)LSTM.")
    parser.add_argument("--bottleneck", type=str, default=None,
                        choices=["dvib", "dropout", "size", None],
                        help="Whether to use a bottleneck in the TreeLSTM.")
    parser.add_argument("--embeddings", type=str, choices=["glove", "random"],
                        help="Type of embeddings to use: random | glove",
                        default="random")
    parser.add_argument("--tensorboard", type=str)

    # training arguments
    parser.add_argument("--epochs", default=15, type=int,
                        help="number of total epochs to run")
    parser.add_argument("--batchsize", default=25, type=int,
                        help="batchsize for optimizer updates")
    parser.add_argument("--lr", default=0.001, type=float, metavar="LR",
                        help="initial learning rate")
    parser.add_argument("--disable_tqdm", action="store_true",
                        help="Whether to use TQDM in command line or not.")
    parser.add_argument("--train_nodes", type=str, choices=["all", "root"],
                        default="root",
                        help="Whether to train using non-root nodes.")
    parser.add_argument("--beta", type=float, default=0,
                        help="Impact of the KL loss component in training.")
    parser.add_argument("--no_validation", action="store_true")

    # miscellaneous options
    parser.add_argument("--seed", default=1, type=int,
                        help="random seed (default: 1)")
    parser.add_argument("--no_saving", action="store_true")

    args = parser.parse_args()
    return args
