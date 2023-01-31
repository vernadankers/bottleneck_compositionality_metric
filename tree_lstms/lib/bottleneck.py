import torch
import torch.nn as nn

# Code from github.com/XiangLi1999/syntactic-VIB/blob/master/src/gaussian_tag.py
SMALL = 1e-08


class Bottleneck(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden2mean = nn.Linear(hidden_dim, hidden_dim)
        self.hidden2std = nn.Linear(hidden_dim, hidden_dim)
        self.r_mean = nn.Parameter(torch.randn(1, hidden_dim))
        self.r_std = nn.Parameter(torch.randn(1, hidden_dim))

    def forward(self, hidden_state, training_mode):
        """
        Draw a sample and compute the KL divergence for a given hidden state.
        Modelled after
        github.com/XiangLi1999/syntactic-VIB/blob/master/src/gaussian_tag.py

        Args:
            - hidden_state (FloatTensor): hidden state coming from the TreeLSTM
            - training_mode (str): inference | training
        Returns:
            - transformed hidden state (FloatTensor)
            - KL divergence to normal distribution (FloatTensor)
        """
        mean = self.hidden2mean(hidden_state)
        std = self.hidden2std(hidden_state)
        cov = std * std + SMALL
        sample = self.get_sample(mean, cov, training_mode)

        mean_r = self.r_mean.expand(hidden_state.shape[0], -1)
        std_r = self.r_std.expand(hidden_state.shape[0], -1)
        cov_r = std_r * std_r + SMALL
        kl_div = self.kl_div(mean, cov, mean_r, cov_r)
        return sample, kl_div

    def get_sample(self, mean, cov, training_mode):
        """
        Reparametrisation trick.
        Modelled after
        github.com/XiangLi1999/syntactic-VIB/blob/master/src/gaussian_tag.py

        Args:
            - mean (torch.FloatTensor)
            - std (torch.FloatTensor)
            - training_mode (str): training | inference
        Returns:
            - torch.FloatTensor
        """
        if training_mode == "inference":
            return mean
        return mean + torch.randn(mean.shape) * torch.sqrt(cov)

    def kl_div(self, mean1, cov1, mean2, cov2):
        """
        KL div from
        github.com/XiangLi1999/syntactic-VIB/blob/master/src/gaussian_tag.py
        Args:
            - mean (torch.FloatTensor)
            - cov (torch.FloatTensor)
            - mean of the prior distribution (torch.FloatTensor)
            - cov of the prior distribution (torch.FloatTensor)
        Returns:
            - kl div (torch.FloatTensor)
        """
        bsz, tag_dim = mean1.shape

        cov2_inv = 1 / cov2
        mean_diff = mean1 - mean2

        mean_diff = mean_diff.view(bsz, -1)
        cov1 = cov1.view(bsz, -1)
        cov2 = cov2.view(bsz, -1)
        cov2_inv = cov2_inv.view(bsz, -1)

        temp = (mean_diff * cov2_inv).view(bsz, 1, -1)
        KL = 0.5 * (torch.sum(torch.log(cov2), dim=1)
                    - torch.sum(torch.log(cov1), dim=1) - tag_dim
                    + torch.sum(cov2_inv * cov1, dim=1)
                    + torch.bmm(temp, mean_diff.view(bsz, -1, 1)).view(bsz))
        return KL
