import torch
from LLM4Graph.train.utils import get_metric_from_cfg

def single_gpu_test(model, test_loader, cfg):
    """
        Test the model
    """
    model.eval()
    metric = get_metric_from_cfg(cfg)
    for _, batch in enumerate(test_loader):
        batch = batch.to(cfg.device)
        with torch.no_grad():
            pred, _ = model(batch)
        metric.update(pred, batch.y)
    return metric.compute()
        