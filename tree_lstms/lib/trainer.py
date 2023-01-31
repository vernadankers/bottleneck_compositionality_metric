from tqdm import tqdm
import torch
import numpy as np
from collections import defaultdict


class Trainer():
    """Trainer object that can both train and test the model."""

    def __init__(self, args, vocab, model, metrics, teacher, max_steps):
        self.criterion = torch.nn.MSELoss() if args.dataset == "arithmetic" \
            else torch.nn.NLLLoss()

        # Model, criteria and optimiser
        model_parameters = list(
            filter(lambda p: p.requires_grad, model.parameters()))
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model_parameters, lr=args.lr)
        self.cosine_fn = torch.nn.CosineEmbeddingLoss(reduction='none')
        self.mse = torch.nn.MSELoss(reduction="none")
        self.train_all_nodes = args.train_nodes == "all"
        self.dataset = args.dataset

        # Object to store evaluation metrics
        self.metrics = metrics

        # Params needed for the bottleneck training
        self.teacher = teacher
        self.max_steps = max_steps
        self.steps, self.epoch, self.beta = 0, 0, 0
        self.max_beta = args.beta

    def train(self, epoch, dataset, disable_tqdm):
        """
        Function that loops over the dataset for one epoch.
        Args:
            - epoch (int)
            - dataset (custom Dataset object)
            - disable_tqdm (bool): whether to show progress bar
        Returns:
            - loss (float)
        """
        dataset.shuffle()
        self.model.train()
        self.model.zero_grad()
        indices = torch.randperm(dataset.num_batches, device='cpu')

        desc = f"Training epoch {self.epoch + 1}"
        losses = []
        for i in tqdm(indices, desc=desc, disable=disable_tqdm):
            self.training_mode = "regular"
            self.steps += 1

            # Gradually update beta over the course of training
            self.beta = self.max_beta * (self.steps / self.max_steps)

            # Update model parameters
            batch = dataset.get_batch(i)
            loss = self.train_step(batch)[0]
            loss.backward()
            losses.append(loss.item())
            self.optimizer.step()
            self.model.zero_grad()
        self.epoch += 1
        return np.mean(losses)

    def train_step(self, batch):
        """
        Run the model once based on a given batch.
        Args:
            - batch (tuple containing sentences, trees, idx, labels)
        Returns:
            - loss (float)
            - prediction (model output)
            - hidden states (list of vectors)
            - kl (float)
            - tre_loss (float)
        """
        sentence, tree, input, root_labels = batch
        prediction, hidden_states, kl, all_labels = self.model(
            tree, input, training_mode=self.training_mode)

        # Use root_labels if only training on root, else use all_label
        labels = root_labels if self.dataset != "sentiment" or \
            self.training_mode == "inference" or \
            not self.train_all_nodes else all_labels
        labels = labels.long() if self.dataset == "sentiment" else labels.float()

        # Task loss component of the bottleneck
        loss = self.criterion(prediction, labels)

        # BCM-TT setup, use teacher's states to make hidden states similar
        tre_loss = torch.LongTensor([])
        if self.teacher is not None:
            self.model.eval()
            self.teacher.eval()
            hidden_states = self.model(tree, input, "inference")[1]
            hidden_teacher = self.teacher(tree, input, "inference")[1].detach()
            if self.dataset == "sentiment":
                tgt = torch.LongTensor([1] * hidden_states.shape[0])
                tre_loss = self.cosine_fn(hidden_states, hidden_teacher, tgt)
            else:
                tre_loss = self.mse(hidden_states, hidden_teacher)
            self.model.train()
            loss += tre_loss.mean()

        # Information loss component of the bottleneck
        if self.beta != 0:
            assert kl > 0
            loss += self.beta * kl
        return loss, prediction, hidden_states, kl, tre_loss

    def test(self, dataset, disable_tqdm):
        """
        Evaluate self.model using the given dataset.
        Args:
            - dataset (custom Dataset object)
            - disable_tqdm (bool): whether to show progress bar or not
        Returns:
            - Metrics object
            - model predictions (list of int/float)
            - hidden states (list of vectors)
            - kl (KL divergences)
        """
        stats = defaultdict(list)
        with torch.no_grad():
            desc = f"Testing epoch {self.epoch}"
            for idx in tqdm(range(dataset.num_batches), desc,
                            disable=disable_tqdm):
                self.model.eval()
                self.training_mode = "inference"

                # Get prediction
                batch = dataset.get_batch(idx)
                loss, outputs, hidden, kl, tre_loss = self.train_step(batch)

                # Store outputs and hidden states etc for further analysis
                if self.dataset == "sentiment":
                    outputs = torch.argmax(outputs, dim=-1)
                stats["outputs"].extend(outputs)
                stats["hidden"].extend(hidden.tolist())
                stats["loss"].append(loss.item())
                stats["labels"].extend(batch[3].tolist())
                stats["kl"].append(kl)
                stats["dif_teacher"].extend(tre_loss.tolist())
                stats["trees"].extend(batch[1])

        # Compute a range of success measures dependent on the dataset used
        # and the model used
        metrics = {"loss": np.mean(stats["loss"])}
        if self.model.use_bottleneck:
            metrics.update({"kl": np.mean(stats["kl"])})
        if self.teacher is not None:
            metrics.update({"dif_teacher": np.mean(stats["dif_teacher"])})
        if self.dataset != "arithmetic":
            metrics["accuracy"] = self.metrics.accuracy(
                stats["outputs"], stats["labels"])
            metrics["f1"] = self.metrics.f1(stats["outputs"], stats["labels"])
        else:
            metrics["MSE"] = self.metrics.mse(
                stats["outputs"], stats["labels"])
            metrics["MSE_comp"] = self.metrics.mse(
                stats["outputs"], [eval(str(y)) for y in stats["trees"]])
        return metrics, stats["outputs"], stats["hidden"], stats["kl"]
