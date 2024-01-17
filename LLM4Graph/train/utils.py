import torch
from torch.optim.lr_scheduler import _LRScheduler
from ogb.nodeproppred import Evaluator as NodePropPredEvaluator
from ogb.graphproppred import Evaluator as GraphPropPredEvaluator
from torchmetrics import AUROC, Accuracy, F1Score, MeanAbsoluteError
import torch.optim.lr_scheduler as lr_scheduler

def get_metric_from_cfg(cfg):
    if 'ogbg' in cfg.dataset.name:
        return GraphPropPredEvaluator(cfg.dataset.name)
    elif 'ogbn' in cfg.dataset.name:
        return NodePropPredEvaluator(cfg.dataset.name)
    else:
        if cfg.dataset.eval == 'multiclass-accuracy' or cfg.dataset.eval == 'accuracy':
            metric = Accuracy(task = 'multiclass', num_classes=cfg.model.num_classes)
        elif cfg.dataset.eval == 'binary-accuracy':
            metric = Accuracy(task = 'binary')
        elif cfg.dataset.eval == 'multiclass-auroc' or cfg.dataset.eval == 'auroc':
            metric = AUROC(task='multiclass', num_classes=cfg.model.num_classes)
        elif cfg.dataset.eval == 'binary-auroc':
            metric = AUROC(task='binary')
        elif cfg.dataset.eval == 'macrof1':
            metric = F1Score(average='macro', task='multiclass', num_classes=cfg.model.num_classes)
        elif cfg.dataset.eval == 'microf1':
            metric = F1Score(average='micro', task='multiclass', num_classes=cfg.model.num_classes)
        elif cfg.dataset.eval == 'mae':
            metric = MeanAbsoluteError()
        else:
            raise NotImplementedError
        return metric
        

def get_loss_fn_from_cfg(cfg):
    if cfg.dataset.loss == 'cross-entropy':
        return torch.nn.CrossEntropyLoss()


def get_optim_from_cfg(model, cfg):
    if cfg.train.optim == 'adam':
        return torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)


def get_scheduler_from_cfg(optimizer, cfg):
    if cfg.train.scheduler == 'ExponentialLR':
        return lr_scheduler.ExponentialLR(optimizer, gamma=cfg.train.lr_reduce_factor)
    else:
        return None


class PolynomialDecayLR(_LRScheduler):

    def __init__(self, optimizer, warmup_updates, tot_updates, lr, end_lr, power, last_epoch=-1, verbose=False):
        self.warmup_updates = warmup_updates
        self.tot_updates = tot_updates
        self.lr = lr
        self.end_lr = end_lr
        self.power = power
        super(PolynomialDecayLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self._step_count <= self.warmup_updates:
            self.warmup_factor = self._step_count / float(self.warmup_updates)
            lr = self.warmup_factor * self.lr
        elif self._step_count >= self.tot_updates:
            lr = self.end_lr
        else:
            warmup = self.warmup_updates
            lr_range = self.lr - self.end_lr
            pct_remaining = 1 - (self._step_count - warmup) / (
                self.tot_updates - warmup
            )
            lr = lr_range * pct_remaining ** (self.power) + self.end_lr

        return [lr for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        assert False