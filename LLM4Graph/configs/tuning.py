def params(trial, model, dataset):
    """
        an example, set the range here
    """
    if dataset == 'Cora':
        return 
    {
        'train.lr': trial.suggest_categorical('train.lr', [1e-2, 1e-3]),
        'train.weight_decay': trial.suggest_categorical('train.weight_decay', [1e-5, 5e-4, 0]),
        'model.hidden_dim': trial.suggest_categorical('model.hidden_dim', [32, 64, 128, 256]),
        'model.dropout': trial.suggest_categorical('model.dropout', [0., .1, .5,]),
        'model.nlayer_gnn': trial.suggest_categorical('model.nlayer_gnn', [2, 3])
    }