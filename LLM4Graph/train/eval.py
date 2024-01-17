import torch
from LLM4Graph.train.utils import get_metric_from_cfg

@torch.no_grad()
def single_gpu_test(model, test_loader, cfg, device):
    """
        Test the model
    """
    model.eval()
    metric = get_metric_from_cfg(cfg)
    for _, batch in enumerate(test_loader):
        if cfg.dataset.loader == 'torch':
            X, y = batch 
            X = X.to(device)
            if cfg.dataset.eval != 'mae':
                y = y.long()
            y = y.to(device)
            pred, _ = model(X)
            metric.update(pred.cpu(), y.cpu())
        else:
            batch = batch.to(device)
            pred, _ = model(batch)
            metric.update(pred.cpu(), batch.y.cpu())
    return metric.compute()
        