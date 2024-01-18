## *****************
## Utility functions for training on a single gpu card
## *****************
import torch
from LLM4Graph.train.eval import single_gpu_test
from LLM4Graph.train.utils import get_loss_fn_from_cfg, get_metric_from_cfg

def train(model, train_loader, val_loader, test_loader, optimizer, scheduler, cfg, device):
    """
        Train the model for one epoch
    """
    model.train()
    loss_fn = get_loss_fn_from_cfg(cfg)
    metric = get_metric_from_cfg(cfg)
    for _, batch in enumerate(train_loader):
        optimizer.zero_grad()
        # import ipdb; ipdb.set_trace()
        if cfg.dataset.loader == 'torch':
            ## in torch style, no edge_index and data object
            X, y = batch 
            X = X.to(device)
            if cfg.dataset.eval != 'mae':
                y = y.long()
            y = y.to(device)
            pred, logits = model(X)
            loss = loss_fn(logits, y)
            with torch.no_grad():
                metric.update(pred.cpu(), y.cpu())
        else:
            batch = batch.to(device)
            pred, logits = model(batch)
            loss = loss_fn(logits, batch.y)
            with torch.no_grad():
                metric.update(pred.cpu(), batch.y.cpu())
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
    
    train_metric = metric.compute()
    val_metric = single_gpu_test(model, val_loader, cfg, device)
    test_metric = single_gpu_test(model, test_loader, cfg, device)
    return train_metric, val_metric, test_metric
    

        