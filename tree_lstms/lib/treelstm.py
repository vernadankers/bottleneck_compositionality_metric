import pickle
import torch
import numpy as np
import torch.nn as nn
from .utils import batch, unbatch, prepare_batches
from .bottleneck import Bottleneck


class TreeLSTM(nn.Module):
    """Custom TreeLSTM model equipped with bottlenecks."""

    def __init__(self, vocab_size, input_dim, hidden_dim, num_classes,
                 dropout=0.0, model_type="treelstm", bottleneck=False,
                 embeddings='random', dataset="arithmetic", train_nodes="all",
                 freeze_embeddings=False, beta=0, baseline=False, **kwargs):
        super().__init__()
        self.dataset = dataset
        self.hidden_dim = hidden_dim
        self.train_all_nodes = train_nodes == "all"
        if embeddings == "glove":
            input_dim = 300
        self.embedding = nn.Embedding(vocab_size, input_dim, padding_idx=0)

        # TreeLSTM cell
        self.childsumcell = BinaryTreeLSTMCell(input_dim, hidden_dim)

        # Classification layer
        self.linear1 = nn.Linear(hidden_dim, 100)
        self.linear2 = nn.Linear(100, num_classes)

        # Initialise bottlenecks if needed
        if model_type != "treelstm":
            self.use_bottleneck = True
            self.h_bottleneck = Bottleneck(hidden_dim)
            self.c_bottleneck = Bottleneck(hidden_dim)
        else:
            self.use_bottleneck = False
        self.dropout = nn.Dropout(dropout)

    def init_embeddings(self, vocab):
        """
        Initialise embeddings with Glove for a given vocabulary.
        Args:
            - vocab (custom Vocab object): maps idx-words in .idxToLabel dict
        """
        glove_model = pickle.load(
            open("../data/pickled_data/embeddings.pickle", 'rb'))
        for idx, word in vocab.idxToLabel.items():
            # If the word is in the Glove vocabulary use that vector
            if word in glove_model:
                self.embedding.weight.data[idx, :] = torch.FloatTensor(
                    np.array(glove_model[word]))
            elif word == "<pad>":
                self.embedding.weight.data[idx, :] = torch.zeros(300)
            # Initialise UNK with a mean word embedding
            elif word == "<unk>":
                emb_matrix = torch.FloatTensor(
                    [list(v) for v in glove_model.values()])
                self.embedding.weight.data[idx, :] = emb_matrix.mean(dim=0)
        # Don't optmise the Glove embeddings
        self.embedding.weight.requires_grad = False

    def forward(self, trees, input_ids, training_mode):
        """
        Forward pass of recursive Tree-LSTM.
        Args:
            - trees (Tree objects): batch of trees
            - input_ids (torch.LongTensor): ids of words in flat trees
            - training_mode (str): inference | training
        Returns:
            - prediction (torch.FloatTensor): with 1 (arit) or 5 outputs (sent)
            - hidden states (torch.FloatTensor): 100-dim output vecs
            - kl (float): mean KL term over all nodes
            - all_targets (torch.LongTensor): targets collected from all nodes
        """
        hidden, kl, all_targets = self.forward_treelstm(
            trees, input_ids, training_mode)
        prediction = self.predict(hidden)
        return prediction, self.linear1(hidden), kl, all_targets

    def predict(self, hidden):
        """
        Apply final two-layer classifier, with softmax if doing sentiment.
        Args:
            - hidden (torch.FloatTensor): hidden state
        Returns:
            - Prediction with 1 (arithmetic) output class or 5 (sentiment)
        """
        hidden = self.linear1(hidden)
        output = self.linear2(torch.relu(hidden)).squeeze(-1)
        if self.dataset == "sentiment":
            return torch.log_softmax(output, dim=-1)
        return output

    def forward_treelstm(self, trees, input_ids, training_mode):
        """
        Function that recursive applies the BinaryTreeLSTM cell to nodes.
        Args:
            - trees (Tree objects): batch of trees
            - input_ids (torch.LongTensor): ids of words in flat trees
            - training_mode (str): inference | training
        Returns:
            - hidden states (torch.FloatTensor): output of final nodes OR
              all nodes depending on your experimental setup
            - kl (float): mean KL term over all nodes
            - all_targets (torch.LongTensor): targets collected from all nodes
        """
        # Prepare batches and load embeddings
        embedded_input = self.embedding(input_ids)
        depths, trace = prepare_batches(trees)

        # Now iterate over depths
        collected = dict()
        kl, hidden_states, all_targets = [], [], []
        for depth in range(max(depths)):
            x, y, targets, h, c = batch(
                self.dataset, trace, depth, collected, self.hidden_dim)
            h, c = self.childsumcell.node_forward(embedded_input[x, y], h, c)

            # Apply bottleneck to non-leaf nodes
            if depth > 0:
                h = self.dropout(h)
                c = self.dropout(c)
                if self.use_bottleneck:
                    h, kl_div_h = self.h_bottleneck(h, training_mode)
                    c, kl_div_c = self.c_bottleneck(c, training_mode)
                    kl.append(kl_div_h + kl_div_c)
                hidden_states.append(h)
                all_targets.extend(targets)
            collected = unbatch(self.dataset, x, y, h, c, collected)

        if training_mode == "inference" or not self.train_all_nodes:
            hidden_states = torch.stack(
                [collected[(i, trees[i].idx()[0])][0]
                 for i in range(len(trees))])
        else:
            hidden_states = torch.cat(hidden_states, dim=0)
            all_targets = torch.LongTensor(all_targets)

        kl = -1 if not self.use_bottleneck else torch.mean(torch.cat(kl))
        return hidden_states, kl, all_targets


class BinaryTreeLSTMCell(nn.Module):
    """
    Binary TreeLSTM modelled after https://arxiv.org/pdf/1503.00075.pdf
    """

    def __init__(self, in_dim, mem_dim):
        super().__init__()
        """
        Initialise BinaryTreeLSTMCell.
        Args:
            - in_dim (int): dimension of input vectors
            - mem_dim (int): size of hidden vecs inside of TreeLSTM
        """
        self.ioux = nn.Linear(in_dim, 3 * mem_dim)
        self.iouh1 = nn.Linear(mem_dim, 3 * mem_dim)
        self.iouh2 = nn.Linear(mem_dim, 3 * mem_dim)
        self.fx = nn.Linear(in_dim, mem_dim)
        self.fh11 = nn.Linear(mem_dim, mem_dim)
        self.fh12 = nn.Linear(mem_dim, mem_dim)
        self.fh21 = nn.Linear(mem_dim, mem_dim)
        self.fh22 = nn.Linear(mem_dim, mem_dim)

    def node_forward(self, x, child_h, child_c):
        """
        Compute h(idden state) and c(ell state) of parent based on children.
        Args:
            - x (torch.FloatTensor): input token representation
            - child_h (torch.FloatTensor): hidden states of the children
            - child_c (torch.FloatTensor): memory cell states of the children
        Returns:
            - h (torch.FloatTensor): hidden state of the parent
            - c (torch.FloatTensor): memory cell state of the parent
        """
        # Input output gates
        iou = self.ioux(x) + self.iouh1(child_h[0]) + self.iouh2(child_h[1])
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)

        # Forget gate with off-diagonal parameters
        fx = self.fx(x)
        f0 = torch.sigmoid(self.fh11(child_h[0]) + self.fh12(child_h[1]) + fx)
        f1 = torch.sigmoid(self.fh21(child_h[0]) + self.fh22(child_h[1]) + fx)

        # Compute hidden and memory cell states
        c = i * u + f0 * child_c[0] + f1 * child_c[1]
        h = o * torch.tanh(c)
        return h, c
