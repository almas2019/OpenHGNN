from openhgnn import Experiment
def search_space(trial):
    return {
        "lr": trial.suggest_categorical("lr", [1e-3, 5e-3, 1e-4]),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64]),
        "dropout": trial.suggest_uniform("dropout", 0.0, 0.5),
        'num_layers': trial.suggest_int('num_layers', 2, 3)
    }

experiment = Experiment(model='fastGTN', dataset="tcell_fib", task='node_classification', gpu=-1,early_stopping="False",
                        hpo_search_space=search_space,adaptive_lr_flag='True', hpo_trials=20,max_epoch=100, patience= 20,
                        norm_emd_flag=True, mini_batch_flag=False)
experiment.run()
print(experiment)