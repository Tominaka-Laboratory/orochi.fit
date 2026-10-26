import optuna


def TPE():
    return optuna.samplers.TPESampler(
        multivariate=True, warn_independent_sampling=False, group=True
    )
