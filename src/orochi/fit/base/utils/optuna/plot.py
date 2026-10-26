def plot_optuna_result(study, name, out_path):
    optuna_util.plot_pareto(
        study, out_path + "/" + name + "_pareto_all.png", True
    )
    optuna_util.plot_pareto(
        study, out_path + "/" + name + "_pareto_front.png", False
    )
    # optuna_util.plot_history(study, out_path +"/"+ name + "_history.png")
