#### Helper functions for tuning hyperparameters ####
import optuna 
import torch

def max_trial_callback(study, trial, max_try):
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE or t.state == optuna.trial.TrialState.RUNNING])
    n_total_complete = len([t for t in study.trials])
    if n_complete >= max_try or n_total_complete >= 2 * max_try:
        study.stop()
        torch.cuda.empty_cache()

def delete_failed_trials(study_name: str, storage_url: str):
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    
    failed_trial_ids = [trial._trial_id for trial in study.trials if trial.state == optuna.trial.TrialState.FAIL]

    if not failed_trial_ids:
        print("No failed trials to delete.")
        return

    # Establish a database connection using the storage URL
    engine = optuna.storages.get_storage(storage_url).engine
    connection = engine.connect()

    # Delete failed trials
    for trial_id in failed_trial_ids:
        connection.execute(f"DELETE FROM trials WHERE trial_id = {trial_id};")
    
    connection.close()
    print(f"Deleted {len(failed_trial_ids)} failed trials.")