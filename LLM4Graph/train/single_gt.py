## *****************
## Utility functions for training on a single gpu card
## *****************
from LLM4Graph.train.eval import single_gpu_test
from LLM4Graph.train.utils import get_loss_fn_from_cfg, get_metric_from_cfg, get_optim_from_cfg, get_scheduler_from_cfg

def train(model, train_loader, val_loader, test_loader, optimizer, scheduler, cfg):
    """
        Train the model for one epoch
    """
    loss_fn = get_loss_fn_from_cfg(cfg)
    metric = get_metric_from_cfg(cfg)
    for _, batch in enumerate(train_loader):
        optimizer.zero_grad()
        batch = batch.to(model.device)
        pred, logits = model(batch)
        loss = loss_fn(logits, batch.y)
        metric.update(pred, batch.y)
        loss.backward()
        optimizer.step()
    if scheduler is not None:
        scheduler.step()
    
    train_metric = metric.compute()
    val_metric = single_gpu_test(model, val_loader, cfg)
    test_metric = single_gpu_test(model, test_loader, cfg)
    return train_metric, val_metric, test_metric
    

        