def params(trial, model, dataset):
    """
        an example, set the range here
    """
    if model.lower() == 'nagphormer':
        return {
            'train.lr': trial.suggest_categorical('train.lr', [1e-2, 1e-3]),
            'model.nlayer_gt': trial.suggest_categorical('model.nlayer_gt', [1, 2, 3]),
            'model.gt.pe_dim': trial.suggest_categorical('model.gt.pe_dim', [3, 10, 15]),
            'model.feature_prop_hop': trial.suggest_categorical('model.feature_prop_hop', [3, 7, 10, 16]),
            'model.nhead': trial.suggest_categorical('model.nhead', [1, 8])
        }