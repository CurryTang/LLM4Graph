from LLM4Graph.data.Pygdata import get_data
from LLM4Graph.data.utils import get_loader_from_config
from LLM4Graph.configs.config import cfg, update_cfg
from LLM4Graph.utils.data_utils import seed_everything
from LLM4Graph.train import single_gpu_train, get_optim_from_cfg, get_scheduler_from_cfg
from LLM4Graph.models import get_model
import wandb
import ipdb
import numpy as np
import warnings
import torch
# Suppress all warnings
warnings.filterwarnings("ignore")



def main(cfg):
    data = get_data(cfg)
    seeds = cfg.seeds
    if cfg.dataset.objective == 'maximize':
        best_val_acc = 0
        best_test_acc = 0
    else:
        best_val_acc = 10000
        best_test_acc = 10000
    if hasattr(data, 'x'):
        cfg.model.num_features = data.x.shape[-1]
    cfg.model.num_classes = data.y.max().item() + 1
    model = get_model(cfg)
    optimizer = get_optim_from_cfg(model, cfg)
    scheduler = get_scheduler_from_cfg(optimizer, cfg)
    is_cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda_available else "cpu")
    model = model.to(device)
    best_test_accs = []
    best_val_accs = []
    for s in seeds:
        seed_everything(s)
        early_stop_accum = 0
        train_loader, val_loader, test_loader =  get_loader_from_config(data, cfg, s)
        if cfg.dataset.objective == 'maximize':
            best_val_acc = 0
        else:
            best_val_acc = 10000
        for i in range(cfg.train.num_epochs):
            train_acc, val_acc, test_acc = single_gpu_train(model, train_loader, val_loader, test_loader, optimizer, scheduler, cfg, device)
            if cfg.show_train_details:
                print(f"Epoch {i}: Train Acc: {train_acc}, Val Acc: {val_acc}, Test Acc: {test_acc}")
                wandb.log({'train_acc': train_acc, 'val_acc': val_acc, 'test_acc': test_acc})
            if cfg.dataset.objective == 'maximize':
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    early_stop_accum = 0
                else:
                    early_stop_accum += 1
            else:
                if val_acc < best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    early_stop_accum = 0
                else:
                    early_stop_accum += 1
            if early_stop_accum >= cfg.train.early_stop_patience and cfg.train.early_stop:
                print(f"Early stop at epoch {i}")
                break
            elif optimizer.param_groups[0]['lr'] <= cfg.train.min_lr and cfg.train.min_lr > 0:
                print(f"Minimum lr reached at epoch {i}")
                break
        print(f"Best Val Acc: {best_val_acc}, Best Test Acc: {best_test_acc}")
        wandb.log({'best_val_acc': best_val_acc, 'best_test_acc': best_test_acc})
        best_test_accs.append(best_test_acc)
        best_val_accs.append(best_val_acc)
    avg_test_acc = np.mean(best_test_accs)
    std_test_acc = np.std(best_test_accs)
    print("Average Test Acc: {:.4f} +- {:.4f}".format(avg_test_acc, std_test_acc))
    return np.mean(best_val_accs)

    
    

if __name__ == '__main__':
    cfg = update_cfg(cfg)
    # import ipdb; ipdb.set_trace()
    if cfg.wandb_enable:
        wandb.login(key=cfg.wandb)
        wandb.init(project=cfg.wandb_project, name=cfg.exp_name, config=cfg)
    else:
        wandb.init(mode='disabled')
    main(cfg)