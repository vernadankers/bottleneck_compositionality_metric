import pickle
import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTM(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.name = "lstm"
        self.input_dim = 300
        glove = pickle.load(
            open("../data/pickled_data/embeddings.pickle", 'rb'))
        self.embedding = nn.Embedding(len(glove) + 2, 300,
                                      padding_idx=0)
        self.embedding.weight.requires_grad = False
        self.embedding.weight.data[0, :] = torch.zeros((300))
        self.word_to_index = dict()
        self.embedding.weight.data[1, :] = torch.FloatTensor(
            list(glove.values())).mean(dim=0)
        for i, word in enumerate(glove):
            self.embedding.weight.data[i+2, :] = torch.FloatTensor(glove[word])
            self.word_to_index[word] = i+2
        self.word_to_index["<unk>"] = 1
        self.emb_dropout = nn.Dropout(0.5)
        self.classifier_dropout = nn.Dropout(0.1)

        self.lstm = torch.nn.LSTM(
            300, 300, 2, batch_first=True, bidirectional=True, dropout=0.1)
        self.linear1 = nn.Linear(600, 100)
        self.linear2 = nn.Linear(100, 5)

    def preprocess(self, sentences):
        seq_lengths = torch.LongTensor([len(s.split()) for s in sentences])
        input_ids = [[self.word_to_index.get(w.lower(), 1)
                      for w in s.split()] for s in sentences]
        max_length = torch.max(seq_lengths).item()
        input_ids = [ids + [0] * (max_length - len(ids)) for ids in input_ids]
        if torch.cuda.is_available():
            return torch.LongTensor(input_ids).cuda(), seq_lengths.cuda()
        return torch.LongTensor(input_ids), seq_lengths

    def forward(self, sentences):
        input_ids, seq_lengths = self.preprocess(sentences)
        hidden = self.forward_lstm(self.lstm, input_ids, seq_lengths)
        hidden = self.classifier_dropout(self.linear1(hidden))
        prediction = torch.log_softmax(self.linear2(
            torch.relu(hidden)).squeeze(-1), dim=-1)
        return prediction

    def forward_lstm(self, lstm, input_ids, seq_lengths, embed=True):
        embedded_input = self.emb_dropout(self.embedding(input_ids))
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        unsort_idx = torch.argsort(perm_idx)
        embedded_input = embedded_input[perm_idx]
        packed_input = pack_padded_sequence(
            embedded_input, seq_lengths.cpu(), batch_first=True)
        packed_output, (hn, cn) = lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        output = output[unsort_idx]
        if torch.cuda.is_available():
            seq_lengths = seq_lengths.cuda()
        output = torch.sum(output, dim=1)/seq_lengths[unsort_idx].unsqueeze(-1)
        return output


class Roberta(nn.Module):
    def __init__(self, model_type, **kwargs):
        super().__init__()
        self.name = "transformer"
        self.model = RobertaModel.from_pretrained(
            pretrained_model_name_or_path='roberta-base')
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.linear1 = nn.Linear(768, 100)
        self.linear2 = nn.Linear(100, 5)
        self.classifier_dropout = nn.Dropout(0.1)

    def forward(self, sentences):
        encoded_input = self.tokenizer(
            list(sentences), return_tensors='pt', padding=True)
        for k in encoded_input:
            if torch.cuda.is_available():
                encoded_input[k] = encoded_input[k].cuda()
        hidden = self.model(**encoded_input)[0]
        hidden = self.classifier_dropout(self.linear1(hidden[:, 0, :]))
        prediction = torch.log_softmax(self.linear2(
            torch.relu(hidden)).squeeze(-1), dim=-1)
        return prediction
