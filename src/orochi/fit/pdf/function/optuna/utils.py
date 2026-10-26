def post_processing(study, trial, kwargs, temp_modules, local_keys_list):
    if "tuning" not in list(trial.user_attrs.keys()):
        initial_params = trial.params
        tuning_function = kwargs["tuning", "tuning_function"]
        tuning_key = kwargs["tuning", "key"]
        function = enqueue_trial_tuning_model(
            study,
            trial,
            temp_modules,
            tuning_key,
            tuning_function,
            initial_params,
            report_name="tuning",
        )
        threading.Thread(target=function).start()

    elif "tuning" in list(trial.user_attrs.keys()):
        if "best_params" in list(trial.user_attrs.keys()):
            initial_params = dict(
                **trial.params, **trial.user_attrs["best_params"]
            )
        else:
            initial_params = trial.params

        refine_function = kwargs["refinement", "refine_function"]
        refine_key = kwargs["refinement", "key"]
        function = enqueue_trial_tuning_model(
            study,
            trial,
            temp_modules,
            refine_key,
            refine_function,
            initial_params,
            report_name="refinement",
        )
        threading.Thread(target=function).start()

    # param adjuster==================================================
    if "report_name" not in list(trial.user_attrs.keys()):
        if "adjustment" in list(kwargs.keys()):
            for index, adjustment in enumerate(kwargs["adjustment"]):

                function = adjustment(
                    study,
                    trial,
                    temp_modules,
                    local_keys_list,
                    report_name="adjustment",
                )

                threading.Thread(target=function).start()


def parameter_adjuster(adjuster_key, adjust_function):
    def define_function(study, trial, modules, input_key_list, report_name):
        def running():
            for module_id, orochi_module in enumerate(modules):
                keys = input_key_list[module_id]
                if adjuster_key in keys:
                    is_updated, res = adjust_function(
                        adjuster_key, orochi_module
                    )

                    if np.sum(orochi_module[adjuster_key] - res) == 0.0:
                        is_updated = False

                    if is_updated:
                        fixed_params = make_param_dictionary(
                            orochi_module, module_id, adjuster_key, res
                        )
                        study.enqueue_trial(
                            params=fixed_params,
                            user_attrs={
                                "report_name": report_name,
                                report_name: adjuster_key,
                            },
                            skip_if_exists=True,
                        )

        return running

    return define_function


def parameter_adjuster_based_on_scores(
    adjuster_key, param_scoring_function, adjust_function
):
    def define_function(study, trial, modules, input_key_list, report_name):
        def running():
            for module_id, orochi_module in enumerate(modules):
                keys = input_key_list[module_id]
                if adjuster_key in keys:
                    scores = param_scoring_function(orochi_module)
                    score_dimension = np.size(np.shape(scores))
                    adjust_ids = decide_adjust_index(
                        trial, scores, score_dimension
                    )
                    function = adjust_function(orochi_module)

                    p = adjust_parameters_in_index_list(
                        trial,
                        function,
                        adjust_ids,
                        adjuster_key,
                        score_dimension,
                    )

                    for adjust_id in adjust_ids:
                        fixed_params = next(p)
                        study.enqueue_trial(
                            params=fixed_params,
                            user_attrs={
                                "report_name": report_name,
                                report_name: adjuster_key
                                + "_"
                                + str(adjust_id),
                            },
                            skip_if_exists=True,
                        )

        return running

    return define_function


def adjust_parameters_in_index_list(
    trial, func, adjust_ids, score_key, score_dimension
):
    for adjust_id in adjust_ids:
        fixed_params = {}

        for param_key, param_value in list(trial.params.items()):
            if param_key in list(fixed_params.keys()):
                continue

            unpacked = param_key.split("_")
            module_id = unpacked[0]
            unpacked_key = unpacked[1]
            param_dim = len(unpacked) - 2

            if unpacked_key == score_key:

                if param_dim == 0:
                    fixed_params[param_key] = func()

                elif param_dim == 1:
                    if score_dimension == 1:
                        id1 = int(unpacked[2])
                        if id1 == adjust_id:

                            # proporal from the function
                            prop = func(id1)

                            if prop == None:
                                pass
                            else:
                                fixed_params[param_key] = float(prop)
                                continue
                        else:
                            fixed_params[param_key] = param_value

                elif param_dim == 2:
                    if score_dimension == 1:
                        id1 = int(unpacked[2])
                        if id1 == adjust_id:

                            # proporal from the function
                            prop = func(id1)

                            if isinstance(prop, (list, np.ndarray)):
                                for id2, prop_value in enumerate(prop):
                                    name = (
                                        str(module_id)
                                        + "_"
                                        + str(unpacked_key)
                                        + "_"
                                        + str(id1)
                                        + "_"
                                        + str(id2)
                                    )
                                    fixed_params[name] = float(prop_value)
                                    # print(name, trial.params[name], prop_value)
                                continue
                            else:
                                if prop == None:
                                    pass
                                else:
                                    fixed_params[param_key] = float(prop)
                                    continue

                        else:
                            fixed_params[param_key] = param_value

                    if score_dimension == 2:
                        # this wont match the single value of adjust value. that should be in a list.
                        pass

                else:
                    # this will be made in the future
                    pass
            else:
                fixed_params[param_key] = param_value

        yield fixed_params


def decide_adjust_index(trial, scores, score_dimension):
    if len(list(trial.params.keys())) == 0:
        adjust_ids = []

    elif score_dimension == 0:  # one value object
        adjust_ids = []

    elif score_dimension == 1:  # 1d array
        mu = np.mean(scores)
        sigma = np.std(scores)
        adjust_ids = np.where((scores >= mu))[0]

    elif score_dimension == 2:  # 2D array
        adjust_ids = []

    return adjust_ids


def enqueue_trial_tuning_model(
    study,
    trial,
    temp_modules,
    local_keys_list,
    refine_function,
    initial_params,
    report_name,
):
    def running():
        _, best_params, score = refine_function(
            trial, temp_modules, local_keys_list, initial_params
        )

        params = {}
        param_keys = list(trial.params.keys())
        best_params_keys = list(best_params.keys())

        for key in param_keys:
            if key in best_params_keys:
                params[key] = best_params[key]

        study.enqueue_trial(
            params=params,
            user_attrs={
                "report_name": report_name,
                "score": score,
                "best_params": best_params,
            },
            skip_if_exists=True,
        )

    return running


def optimize_nested_study(score_function, cycles, is_prune=True, **kwargs):
    def running(trial, modules, optimize_keys_list, initial_params):

        def pruning(
            nested_study: optuna.study.Study,
            nested_trial: optuna.trial.FrozenTrial,
        ) -> None:
            if "directions" not in list(kwargs.keys()):
                nested_study.best_value
            else:
                if len(kwargs["directions"]) == 1:
                    trial_value = nested_study.best_value
                else:
                    trial_value = nested_study.best_values

            trial.report(trial_value, nested_trial.number)
            if trial.should_prune():
                nested_study.stop()
                # raise optuna.TrialPruned()

        def trial_logging(
            nested_study: optuna.study.Study,
            nested_trial: optuna.trial.FrozenTrial,
        ) -> None:
            if "directions" not in list(kwargs.keys()):
                best_value = nested_study.best_value
                trial_value = nested_trial.value
            else:
                if len(kwargs["directions"]) == 1:
                    best_value = nested_study.best_value
                    trial_value = nested_trial.value
                else:
                    best_trials = nested_study.best_trials
                    trial_value = nested_trial.values
                    for id, t in enumerate(best_trials):
                        if id == 0:
                            best_value = t.values
                        else:
                            if t.values[0] < best_value[0]:
                                best_value = t.values

            report = [
                "    ",
                nested_trial.datetime_start,
                "trial:",
                trial.number,
                nested_trial.number,
                "value:",
                trial_value,
                "nested best value:",
                best_value,
            ]
            if "log_file" in list(kwargs.keys()):
                print(*report, file=open(kwargs["log_file"], "a"))
            else:
                print(report)

        def nested_objective(
            modules, optimize_keys_list, score_function, kwargs
        ):
            input_modules = copy.deepcopy(modules)
            input_key_list = copy.deepcopy(optimize_keys_list)

            def return_score(nested_trial):
                temp_modules = copy.deepcopy(input_modules)
                local_keys_list = copy.deepcopy(input_key_list)

                if nested_trial.number > 0:
                    for module_id, orochi_module in enumerate(temp_modules):
                        optimize_keys = local_keys_list[module_id]
                        temp_modules[module_id] = suggest_params(
                            nested_trial,
                            orochi_module,
                            optimize_keys,
                            module_id,
                            kwargs,
                        )

                score = score_function(temp_modules)

                final_params = get_current_param_from_modules(
                    temp_modules, local_keys_list
                )
                # nested_trial.set_user_attr("params", final_params)
                nested_trial.set_system_attr("fixed_params", final_params)

                return score

            return return_score

        if "sampler" not in list(kwargs.keys()):
            sampler = optuna.samplers.CmaEsSampler(
                x0=initial_params,
                independent_sampler=optuna.samplers.TPESampler(
                    multivariate=True,
                    warn_independent_sampling=False,
                    group=True,
                ),
                warn_independent_sampling=False,
            )
        else:
            sampler = kwargs["sampler"]

        if is_prune:
            callbacks = [pruning]
        else:
            callbacks = []

        nested_study = optuna.create_study(
            sampler=sampler,
            directions=kwargs["directions"],
            storage="sqlite:///:memory:",
        )

        nested_study.enqueue_trial(params=initial_params, skip_if_exists=True)
        nested_study.optimize(
            nested_objective(
                modules, optimize_keys_list, score_function, kwargs
            ),
            n_trials=cycles,
            n_jobs=1,
            # callbacks = [trial_logging, pruning]
            callbacks=callbacks,
        )
        best_params = (
            nested_study.best_params
        )  # best_trial.user_attrs["params"]
        modules = update(modules, best_params, score_function)
        return modules, best_params, nested_study.best_value

    return running
