import copy
import logging
import queue

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger('mylogger')


class CrossEntropyLoss(nn.CrossEntropyLoss):

    def __init__(self, reduction="none"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target, weight=None):
        if weight is None:
            weight = self.weight
        return F.cross_entropy(input, target, weight=weight, reduction=self.reduction)


class CDANLoss(nn.Module):
    ''' Ref: https://github.com/thuml/CDAN/blob/master/pytorch/loss.py
    '''

    def __init__(self, use_entropy=True, coeff=1):
        super(CDANLoss, self).__init__()
        self.use_entropy = use_entropy
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.coeff = coeff
        self.entropy_loss = EntropyLoss(coeff=1., reduction='none')

    def forward(self, ad_out, softmax_output=None, coeff=1.0, dc_target=None):
        batch_size = ad_out.size(0) // 2
        dc_target = torch.cat((torch.ones(batch_size), torch.zeros(batch_size)), 0).float().to(ad_out.device)
        loss = self.criterion(ad_out.view(-1), dc_target.view(-1))

        if self.use_entropy:
            entropy = self.entropy_loss(softmax_output)
            entropy.register_hook(grl_hook(coeff))
            entropy = 1 + torch.exp(-entropy)
            source_mask = torch.ones_like(entropy)
            source_mask[batch_size:] = 0
            source_weight = entropy * source_mask
            target_mask = torch.ones_like(entropy)
            target_mask[:batch_size] = 0
            target_weight = entropy * target_mask
            weight = source_weight / torch.sum(source_weight).detach().item() + \
                     target_weight / torch.sum(target_weight).detach().item()
            return self.coeff * torch.sum(weight * loss) / torch.sum(weight).detach().item()
        else:
            return self.coeff * torch.mean(loss.squeeze())


class EntropyLoss(nn.Module):
    ''' Ref: https://github.com/thuml/CDAN/blob/master/pytorch/loss.py
    '''

    def __init__(self, coeff=1., reduction='mean'):
        super().__init__()
        self.coeff = coeff
        self.reduction = reduction

    def forward(self, input):
        epsilon = 1e-5
        entropy = -input * torch.log(input + epsilon)
        entropy = torch.sum(entropy, dim=1)
        if self.reduction == 'none':
            return entropy
        return self.coeff * entropy.mean()


class MMDLoss(nn.Module):
    def compute_paired_dist(self, A, B):
        bs_A = A.size(0)
        bs_T = B.size(0)
        feat_len = A.size(1)

        A_expand = A.unsqueeze(1).expand(bs_A, bs_T, feat_len)
        B_expand = B.unsqueeze(0).expand(bs_A, bs_T, feat_len)
        dists = (((A_expand - B_expand)) ** 2).sum(2)
        return dists

    def gamma_estimation(self, dist):
        dist_sum = (
                torch.sum(dist["ss"]) + torch.sum(dist["tt"]) + 2 * torch.sum(dist["st"])
        )
        bs_S = dist["ss"].size(0)
        bs_T = dist["tt"].size(0)
        N = bs_S * bs_S + bs_T * bs_T + 2 * bs_S * bs_T - bs_S - bs_T
        gamma = dist_sum.item() / N
        return gamma

    def compute_kernel_dist(self, dists, gamma, kernel_num=5, kernel_mul=2):
        base_gamma = gamma / (kernel_mul ** (kernel_num // 2))
        gamma_list = [base_gamma * (kernel_mul ** i) for i in range(kernel_num)]
        gamma_tensor = (torch.tensor(gamma_list)).to(self.device)

        eps = 1e-5
        gamma_mask = (gamma_tensor < eps).type(torch.cuda.FloatTensor)
        gamma_tensor = (1.0 - gamma_mask) * gamma_tensor + gamma_mask * eps
        gamma_tensor = gamma_tensor.detach()

        dists = dists.unsqueeze(0) / gamma_tensor.view(-1, 1, 1)
        upper_mask = (dists > 1e5).type(torch.cuda.FloatTensor).detach()
        lower_mask = (dists < 1e-5).type(torch.cuda.FloatTensor).detach()
        normal_mask = 1.0 - upper_mask - lower_mask
        dists = normal_mask * dists + upper_mask * 1e5 + lower_mask * 1e-5
        kernel_val = torch.sum(torch.exp(-1.0 * dists), dim=0)
        return kernel_val

    def kernel_layer_aggregation(self, dist_layers, gamma_layers):
        kernel_dist = {}
        dists = dist_layers[0]
        gamma = gamma_layers[0]

        if len(kernel_dist.keys()) == 0:
            kernel_dist = {
                key: self.compute_kernel_dist(dists[key], gamma)
                for key in ["ss", "tt", "st"]
            }

        kernel_dist = {
            key: kernel_dist[key] + self.compute_kernel_dist(dists[key], gamma)
            for key in ["ss", "tt", "st"]
        }
        return kernel_dist

    def mmd(self, source, target):
        """
        Computes Maximum Mean Discrepancy
        """
        dists = {}
        dists["ss"] = self.compute_paired_dist(source, source)
        dists["tt"] = self.compute_paired_dist(target, target)
        dists["st"] = self.compute_paired_dist(source, target)

        # import pdb; pdb.set_trace();
        dist_layers, gamma_layers = [], []
        dist_layers += [dists]
        gamma_layers += [self.gamma_estimation(dists)]

        kernel_dist = self.kernel_layer_aggregation(dist_layers, gamma_layers)
        mmd = (
                torch.mean(kernel_dist["ss"])
                + torch.mean(kernel_dist["tt"])
                - 2.0 * torch.mean(kernel_dist["st"])
        )
        return mmd
