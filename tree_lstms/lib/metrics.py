from copy import deepcopy
import torch
import sklearn


class Metrics():
    def __init__(self):
        return

    def mse(self, predictions, labels):
        x = torch.FloatTensor(deepcopy(predictions))
        y = torch.FloatTensor(deepcopy(labels))
        return torch.mean((x - y) ** 2)

    def accuracy(self, predictions, labels):
        predictions = [round(x.item()) for x in predictions]
        return sklearn.metrics.accuracy_score(labels, predictions)

    def f1(self, predictions, labels):
        predictions = [x.item() for x in predictions]
        return sklearn.metrics.f1_score(labels, predictions, average="macro")
