from main import main
from LLM4Graph.configs.config import cfg, update_cfg, update_yacs_config
from LLM4Graph.configs.tuning import params 
from LLM4Graph.utils.tuner import delete_failed_trials, max_trial_callback
import optuna
import logging 
import sys
import wandb


def sweep(cfg):
    study_name = f"{cfg.dataset.name}_{cfg.model.name}"
    delete_failed_trials(study_name, cfg.optuna_db)
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    # study_name = get_current_time_as_filename()
    study = optuna.create_study(study_name=study_name, direction=cfg.dataset.objetive, 
                                storage=cfg.optuna_db, load_if_exists=True)
    param_f = params
    study.optimize(
        lambda trial: sweep_run(
            trial, cfg, param_f
        ), catch=(RuntimeError,), n_trials=cfg.n_trials, 
        callbacks=[lambda study, trial: max_trial_callback(study, trial, cfg.n_trials)], show_progress_bar=True, gc_after_trial=True
    )

    best_cfg = update_yacs_config(cfg, study.best_params)

    print("Best run:")
    main(best_cfg)


def sweep_run(trial, cfg, params_f):
    params = params_f(trial)
    cfg = update_yacs_config(cfg, params)
    best_val = main(cfg)
    return best_val 
    



if __name__ == '__main__':
    cfg = update_cfg(cfg)
    if cfg.wandb_enable:
        wandb.login(key=cfg.wandb)
        wandb.init(project=cfg.wandb_project, name=cfg.exp_name, config=cfg)
    else:
        wandb.init(mode='disabled')
    sweep(cfg)