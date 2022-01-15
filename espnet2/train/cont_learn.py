from copy import deepcopy

import torch
from torch import nn
from torch.autograd import Variable
import torch.utils.data
import numpy as np
from espnet2.torch_utils.device_funcs import to_device
import logging

def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)

def pad2list(padded_seq, lengths):
    return torch.cat([padded_seq[i, 0:lengths[i]] for i in range(padded_seq.size(0))])

def softmax(X):
    return np.exp(X) / np.tile(np.sum(np.exp(X), axis=1)[:, None], (1, X.shape[1]))

def compute_fer(x, l):
    x = softmax(x)
    preds = np.argmax(x, axis=1)
    err = (float(preds.shape[0]) - float(np.sum(np.equal(preds, l)))) * 100 / float(preds.shape[0])
    return err

class CL(object):
    def __init__(self, model: nn.Module, dataset: torch.utils.data.DataLoader, batch_size, seq_len = 512, num_gpu = 0, type = None):

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_gpu = num_gpu

        self.type = type
        # wca: Weight Constraint Adaptation, ewc: Elastic Weight Consolidation, lwf: Learning Without Forgetting (Soft KL Divergence)
        if self.type == 'ewc' or self.type == 'wca':
            self.model = deepcopy(model)
            self.data_domain = dataset
            self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
            self._means = {}
            for n, p in deepcopy(self.params).items():
                self._means[n] = p.data
        if self.type == 'lwf':
            self.model = deepcopy(model)

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p
        for iiter, (_, batch) in enumerate(self.data_domain): 
            self.model.zero_grad()
            batch = to_device(batch, "cuda" if self.num_gpu > 0 else "cpu")
            retval = self.model(**batch)
            loss, stats, weight = retval
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    precision_matrices[n].data += p.grad.data ** 2

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def update_model_weights(self, model: nn.Module):
        if self.type == 'wca':
            self.model = model
        if self.type == 'ewc':
            self.model = model
            self._precision_matrices = self._diag_fisher()

    def penalty(self, model: nn.Module, data = None):
        loss = 0

        if self.type == 'wca':
            for n, p in model.named_parameters():
                _loss = (p - self._means[n]) ** 2
                loss += _loss.sum()

        if self.type == 'ewc':
            for n, p in model.named_parameters():
                _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
                loss += _loss.sum()

        if self.type == 'lwf':
            self.model.train()
            retval = self.model(**data)
            loss, stats, weight = retval

        return loss