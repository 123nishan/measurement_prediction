import torch
print(torch.cuda.is_available())


#
# study=optuna.create_study(direction='maximize',sampler=optuna.sampler.TPESampler())
# study.optimize(objective,n_trials=30)
