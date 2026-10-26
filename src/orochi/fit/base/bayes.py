import numpy as np
try:
    import optuna
except ImportError:
    optuna = None
# import threading
try:
    from joblib import parallel_backend
except ImportError:
    parallel_backend = None
# import scipy.interpolate as intp
import warnings
import copy
import os

warnings.simplefilter("ignore")
from .params import optuna as default
from .base_optimizer import BaseOptimizer


class BayesOptimizer(BaseOptimizer):
    """Performs Bayesian optimization using the Optuna library for Orochi modules.

    This class extends `BaseOptimizer` to implement Bayesian optimization,
    leveraging the powerful features of the `Optuna` framework. It enables
    finding optimal parameters for one or more Orochi modules (e.g., simulators,
    phases) by efficiently exploring the parameter space based on past trial
    results.

    It manages the setup of an Optuna study, defines the objective function
    that interacts with Orochi modules, executes the optimization, and
    propagates the best-found parameters back to the original modules.

    Attributes
    ----------
    study_name : str
        The name of the Optuna study. This attribute directly holds the value
        from `self["study_name"]` and is used to identify the Optuna study
        in its storage backend.
    sampler : optuna.samplers.BaseSampler or None
        The Optuna sampler instance, responsible for suggesting new parameters
        for each trial (e.g., `optuna.samplers.TPESampler`). It is initialized
        within the `set_study` method based on `self["sampler"]`.
    storage : str or None
        The storage URL for the Optuna study (e.g., "sqlite:///example.db" or
        "mysql+pymysql://user:pass@host:port/dbname"). This defines where the
        study's trials and results are persistently stored. It is initialized
        within the `set_study` method based on `self.par["database_type"]`
        and related database connection parameters.
    pruner : optuna.pruners.BasePruner or None
        The Optuna pruner instance, used to stop unpromising trials early
        (e.g., `optuna.pruners.HyperbandPruner`). It is initialized within
        the `set_study` method.
    study : optuna.study.Study or None
        The main Optuna study object, which orchestrates the optimization process.
        It manages trials, handles parameter suggestions, and stores results.
        It is created or loaded in the `set_study` method.
    score_function : callable or None
        A callable that calculates the objective value (score) for a given set
        of simulated results against observed data. This function is typically
        set up by the `BaseOptimizer`'s `__call__` method. It takes `obsx`, `obsy`,
        `simx_attr`, `simy_attr` as initialization arguments and returns a
        further callable that takes a list of Orochi simulator modules as input.
    obsx : numpy.ndarray or None
        The observed independent variable data (x-axis) used for fitting. This
        data is typically loaded and filtered (based on `fit_min` and `fit_max`)
        by the `BaseOptimizer` during its `__call__` execution.
    obsy : numpy.ndarray or None
        The observed dependent variable data (y-axis) corresponding to `obsx`.
        Similar to `obsx`, it's processed by `BaseOptimizer`.
    simulators : list of OrochiBase
        A list of top-level Orochi simulator modules that are subject to optimization.
        These modules (and their nested modules) contain the parameters to be fitted.
        This attribute is populated by `BaseOptimizer`.
    best_params : dict or None
        A dictionary storing the parameter names and their corresponding values
        from the best trial found by Optuna. These parameters are applied back
        to the Orochi modules after optimization.
    best_score : float or None
        The primary goodness-of-fit score obtained from the `best_params`.
        This is extracted from `self.best_scores` for convenience.
        This attribute is populated by `_summarize_scores`.
    best_scores : dict
        A dictionary storing various goodness-of-fit scores for the best fit,
        including the primary score (keyed by `self.par["score_type"]`) and
        the Optuna `trial_number` of the best trial.
        Populated by `_summarize_scores`.
    fit_params : dict
        An internal dictionary that maps simplified parameter names (used in Optuna trials)
        to their full details, including the Orochi module object, the parameter's
        address within that module, its initial value, and its min/max bounds.
        This is crucial for `_suggest_params_to_modules` and `_summarize_params_and_update_modules`.
        Populated by `BaseOptimizer`.
    callbacks : list
        A list of `optuna.integration.callbacks.BaseCallback` objects that
        are executed at various stages of the Optuna optimization process.
        This list can be configured by the user through `self.set_callbacks`
        (though not explicitly shown in the provided code snippet, it's a
        common pattern in Orochi).

    See Also
    --------
    orochi.base.core.OrochiBase : Base class for handling parameters within the Orochi framework.
    orochi.optimizer.base_optimizer.BaseOptimizer : The base class for optimizers in Orochi, providing common functionalities.
    optuna : The Optuna library for hyperparameter optimization, the core of this optimizer.

    Notes
    -----
    The `fit_min` and `fit_max` parameters in `par` define the range of the
    observed data that should be used for fitting. While not explicitly shown
    in this class, the broader Orochi framework often employs an `r_extension`
    constant to ensure that simulator modules calculate data over a slightly
    wider range than the observed data's `fit_min` and `fit_max`. This practice
    helps prevent interpolation issues at the boundaries when comparing
    simulated data to observed data.
    """

    def __init__(self):
        """Initializes the Optimizer class for Bayesian Optimization using Optuna.

        This constructor sets up the default parameters for the optimizer,
        initializes the base `BaseOptimizer` class, and sets up internal attributes
        required for managing the Optuna study, modules, and optimization parameters.
        """
        initial_par = getattr(self, "_par", getattr(self, "par", None))
        if initial_par is not None:
            par = copy.deepcopy(default.par)
            par.update(initial_par)
            self._par = par
        else:
            self._par = copy.deepcopy(default.par)

        key_info = copy.deepcopy(default.par_key_info)
        initial_key_info = getattr(self, "_par_key_info", getattr(self, "par_key_info", None))
        if initial_key_info is not None:
            key_info.update(initial_key_info)
        self._par_key_info = key_info

        super().__init__()
        # optuna.logging.disable_default_handler() # Uncomment if Optuna logs are too verbose

        self._name = "fit.bayes"
        self._id = "3"  # Unique identifier for this optimizer module
    
        self.sampler = None
        self.storage = None
        self.pruner = None
        self.study = None
        self.score_function = None # This attribute is inherited from BaseOptimizer, but re-initialized here.

        self.callbacks = []


    def __call__(self, force: bool = False):
        """Executes the Bayesian optimization process.

        This method orchestrates the entire optimization workflow by calling
        a sequence of internal methods:
        1. Calls `super().__call__(force=force)`: This sets up initial module parameters,
           loads and filters observed data (`obsx`, `obsy`), and initializes
           the `score_function` based on `self.par` settings.
        2. Calls `self.set_study()`: Configures and initializes the Optuna study
           including sampler, storage, pruner, and study name.
        3. Calls `self.fit_optuna()`: Runs the core Optuna optimization loops,
           executing trials and evaluating the objective function.
        4. Calls `self.end_process_optuna()`: Finalizes the optimization by
           retrieving the best parameters and updating the Orochi modules.
        5. Calls `self.report()`: Generates and saves a summary report of the
           optimization results.
        6. Calls `self.save_best_result()`: Saves the simulated data from the
           best fit for later analysis.
        """

        super().__call__(force=force)

        self.set_study()
        self.fit_optuna()
        self.end_process_optuna()
        self.report()
        self.save_best_result()
                        
    def set_study(self):
        """Sets up the Optuna study.

        This method configures the Optuna study's name (`self.study_name`), sampler,
        storage backend, and pruner based on the parameters defined in `self.par`.
        It dynamically constructs the storage URL for SQLite or MySQL databases.
        It then either loads an existing study (if `self["continue_study"]` is True)
        or creates a new one, setting the optimization directions.

        Raises
        ------
        ValueError
            If an unsupported sampler type is specified in `self["sampler"]` or
            an unsupported database type is specified in `self["database_type"]`.
        """

        # Set sampler
        if self["sampler"] == "TPE":
            self.sampler = optuna.samplers.TPESampler(
                multivariate=True, warn_independent_sampling=False, group=True
            )
        elif self["sampler"] == "CMA":
            self.sampler = optuna.samplers.CmaEsSampler(
                warn_independent_sampling=False
            )
        else:
            raise ValueError(f"Unsupported sampler: {self['sampler']}")

        # Set storage
        if self["database_type"] == "sqlite":
            if len(self.work_dir) == 0:
                self.storage = f"sqlite:///optuna/{self.study_name}.db"
            else:
                if not os.path.exists(self.work_dir):
                    os.makedirs(self.work_dir, exist_ok=True)
                self.storage = f"sqlite:///{self.work_dir}/optuna.db"

        elif self["database_type"] == "journal":
            from pathlib import Path
            from optuna.storages import JournalStorage
            from optuna.storages.journal import JournalFileOpenLock
            try:
                from optuna.storages.journal import JournalFileStorage
            except ImportError:
                from optuna.storages.journal import JournalFileBackend as JournalFileStorage

            try:
                from optuna.storages.journal import JournalFileOpenLock
            except ImportError:
                from optuna.storages.journal._file import JournalFileOpenLock                

            if len(self.work_dir) == 0:
                raw_path = f"optuna/{self.study_name}.log"
            else:
                if not os.path.exists(self.work_dir):
                    os.makedirs(self.work_dir, exist_ok=True)
                raw_path = f"{self.work_dir}/optuna.log"

            clean_path = str(Path(raw_path).resolve())
            lock = JournalFileOpenLock(clean_path)
            file_storage = JournalFileStorage(clean_path, lock_obj=lock)
            self.storage = JournalStorage(file_storage)

        elif self["database_type"] == "mysql+pymysql":
            host = self["database_host"]
            port = self["database_port"]
            user = self["database_user"]
            password = self["database_password"]
            self.storage = f"mysql+pymysql://{user}:{password}@{host}:{port}/{self.study_name}"
            
        else:
            raise ValueError(
                f"Unsupported database type: {self['database_type']}"
            )

        # Set pruner
        # Currently hardcoded to HyperbandPruner, but could be made configurable
        self.pruner = None #optuna.pruners.HyperbandPruner()

                # Create or load study
        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            sampler=self.sampler,
            # pruner=self.pruner,
            load_if_exists=self["continue_study"],
            directions=self["directions"],
        )
        self.study.set_user_attr("early_stop", False)

    def fit_optuna(self):
        """Performs the Optuna optimization.

        This method initiates the optimization process by first enqueuing the
        current parameter values (as defined in `self.fit_params`) as the
        initial trial in the Optuna study. It then calls Optuna's `optimize`
        method to run the specified number of trials. The optimization can
        run in parallel using multiprocessing if `self["n_jobs"]` is greater
        than 1, or sequentially otherwise. The `_objective_wrapper` function
        serves as the objective for Optuna, and any configured `callbacks`
        are applied during the optimization.
        """
        # Enqueue the current parameters as the first trial
        params = {}
        for fit_name, info in self.fit_params.items():
            val = info["value"]
            if isinstance(val, (list, tuple)) and len(val) == 1:
                val = val[0]
            params[fit_name] = val

        self.study.enqueue_trial(params=params, skip_if_exists=True)

        if self["n_jobs"] > 1:
            if parallel_backend is None:
                raise ImportError("joblib is required for parallel optimization with n_jobs > 1. Please install joblib via 'pip install joblib'.")
            with parallel_backend("multiprocessing"):
                self.study.optimize(
                    self._objective_wrapper(self.study),
                    n_trials=self["n_trials"],
                    n_jobs=self["n_jobs"],
                    callbacks=self.callbacks,
                )
        else:
            self.study.optimize(
                self._objective_wrapper(self.study),
                n_trials=self["n_trials"],
                callbacks=self.callbacks,
            )

    def _objective_wrapper(self, study):
        """Creates the objective function for Optuna.

        This method acts as a factory, returning a nested function (`objective`)
        that Optuna will call for each trial. The `objective` function encapsulates
        the core logic for a single optimization trial:
        1. **Parameter Reset**: It resets all optimizable parameters in the
           Orochi modules to their initial configured state (`info["value"]`).
           This ensures a consistent starting point for each trial before
           Optuna's suggestions are applied.
        2. **Clear Changed Flags**: It clears internal "changed parameter" flags
           within all simulator modules and their nested components. This is
           critical for Orochi's dependency tracking to correctly re-evaluate
           module outputs when new parameters are set.
        3. **Suggest Parameters**: It calls `_suggest_params_to_modules` to
           receive new parameter values from the current Optuna `trial` object
           (e.g., `trial.suggest_float`, `trial.suggest_int`) and applies these
           values to the corresponding Orochi module parameters.
        4. **Execute Simulators**: It then calls each top-level simulator module
           (`simulator()`) to trigger the generation of simulated data based on
           the newly suggested parameters.
        5. **Calculate Score**: Finally, it calculates the objective value (score)
           using the `score_function` that was set up, comparing the simulated
           data against the observed data (`self.obsx`, `self.obsy`). This score
           is returned to Optuna for optimization.

        Parameters
        ----------
        study : optuna.study.Study
            The Optuna study object. This is passed to the wrapper for potential
            future use (e.g., accessing study-level attributes within the objective).

        Returns
        -------
        callable
            The objective function (`objective`) to be optimized by Optuna.
            This function has the signature `objective(trial)`.
        """

        def objective(trial):
            # Reset modules to their initial state for a fresh trial
            # This ensures that each trial starts with a clean slate before applying new suggestions.
            for n, simulator in enumerate(self.simulators):
                simulator.par = copy.deepcopy(self.init_params[f"{n}_0"])
                simulator._clear_changed_parameters()

                # Initialize parameters for nested modules first
                for m, module_attr in enumerate(simulator["nest_modules"]):
                    module = getattr(simulator, module_attr)
                    module.par = copy.deepcopy(self.init_params[f"{n}_{m+1}"])
                    module._clear_changed_parameters()

            res = {"N_indep":0}

            # Propose new parameters for the current trial using Optuna's suggestions
            self._suggest_params_to_modules(trial)
            for simulator in self.simulators:
                simulator()
                res["N_indep"] += simulator["N_indep"]

            func = self.score_function(obsx=self.obsx, 
                                        obsy=self.obsy,
                                        simx_attr=self["simx_attr"],
                                        simy_attr=self["simy_attr"],
                                        trial=trial
                                        )
            score = func(self.simulators)
            res[self["score_function"]] = float(score)

            k = len(trial.params)
            n_indep = res["N_indep"]

            # --- Guard against division by zero or negative degrees of freedom ---
            if n_indep > k:
                res["adj"] = score * np.sqrt((n_indep - 1.0) / (n_indep - k))
            else:
                res["adj"] = np.inf  # Inf when parameter count exceeds independent data points

            # Compute BIC (assuming score > 0)
            if score > 0 and n_indep > 0:
                res["bic"] = n_indep * np.log(score) + k * np.log(n_indep)
            else:
                res["bic"] = np.inf

            for key, val in res.items():
                trial.set_user_attr(key, float(val))

            return res.get(self["score_type"], score) if n_indep > 0 else score

        return objective

    def _suggest_params_to_modules(self, trial, **kwargs):
        """Helper method to suggest parameters for a single module.

        This method iterates through all optimizable parameters registered in
        `self.fit_params`. For each parameter, it uses the Optuna `trial` object's
        suggestion methods (`trial.suggest_float` or `trial.suggest_int`) to
        propose a new value within the parameter's defined `min` and `max` bounds.
        The newly suggested value is then directly applied to the corresponding
        parameter within the Orochi module.

        Parameters
        ----------
        trial : optuna.trial.Trial
            The current Optuna trial object, which provides methods for suggesting
            new parameter values.
        **kwargs : dict, optional
            Additional keyword arguments. Currently not utilized in this method
            but included for API flexibility.
        """
        for name, info in self.fit_params.items():
            module = info["module"]
            init_value = info["value"]
            address = info["address"]

            if module._check_if_address_constrained(address):
                continue

            if "candidates" in info:
                val = trial.suggest_categorical(name, info["candidates"])
                if isinstance(val, (list, tuple)) and len(val) == 1:
                    val = val[0]
                module[address] = val 
            elif "opt_func" in info:
                module[address] = info["opt_func"](trial)            
            elif isinstance(init_value, float):
                min_val = info["min"]
                max_val = info["max"]
                module[address] = trial.suggest_float(name, min_val, max_val)
            elif isinstance(init_value, int):
                min_val = info["min"]
                max_val = info["max"]
                module[address] = trial.suggest_int(name, min_val, max_val)


    def end_process_optuna(self):
        """Finalizes the Bayesian optimization process.

        This method is called once the Optuna optimization (`fit_optuna`) has
        completed all its trials. It performs the following critical steps:
        1. Calls `_get_best_params_from_study()`: Retrieves the set of parameters
           that yielded the best objective value from the Optuna study.
        2. Calls `_summarize_params_and_update_modules()`: Applies these best
           parameters back to the original Orochi modules, ensuring that the
           modules are configured with the optimized values. It also populates
           `self.best_params`.
        3. Calls `_summarize_scores()`: Calculates and stores the final
           goodness-of-fit scores for the best fit, populating `self.best_scores`
           and `self.best_score`.
        """
        best_params, best_scores = self._get_best_params_from_study()
        self.best_scores = best_scores
        self._summarize_params_and_update_modules(best_params)

    def call_back_best_params(self):
        self.end_process_optuna()

    def _summarize_params_and_update_modules(self, params):
        """Sets parameter values from the best Optuna trial into the Orochi modules.

        This method takes a dictionary of optimized parameter `params` (keyed by
        their Optuna trial names) and systematically applies these values to the
        corresponding parameters within the Orochi modules managed by this optimizer.
        It uses the `self.fit_params` map to find the correct module and its address
        for each optimized parameter. After updating the module parameters, it
        iterates through all top-level simulator modules (`self.simulators`) and
        calls their `__call__` method. This ensures that all internal calculations,
        dependencies, and constraints within the modules are re-applied and
        updated to reflect the new best parameters.
        Finally, it populates the `self.best_params` attribute with the applied
        best parameter values, keyed by their full Orochi address.

        Parameters
        ----------
        params : dict
            A dictionary where keys are parameter names (as used in Optuna trials,
            e.g., "s_module_id/param_name") and values are the optimized parameter values.
        """
        def convert_fit_name_to_module_address(fit_name, prefix, delimiter, module):
            mod_prefix = module["prefix"]
            mod_delimiter = module["address_delimiter"]
            address = fit_name.replace(prefix, mod_prefix)
            address = address.replace(delimiter, mod_delimiter)
            return address

        self.best_params = {}
        self.clear_changed_params()

        for name, value in params.items():
            # Retrieve module and address using the internal fit_params map

            try:
                prefix = self["prefix"]
                delimiter = self["address_delimiter"]
                if delimiter in name:
                    id = name.split(delimiter)[0].strip(prefix)
                    sim_id, nest_id = id.split("_")

                    simulator = self.simulators[int(sim_id)]
                    if nest_id == "0":
                        module = simulator
                    else:
                        module_attr = simulator["nest_modules"][int(nest_id)-1]
                        module = getattr(simulator, module_attr)

                    address = convert_fit_name_to_module_address(name, prefix, delimiter, module)
                    module[address] = value
                    self.best_params[address] = value

            except:
                if name in self.fit_params:
                    module = self.fit_params[name]["module"]
                    address = self.fit_params[name]["address"]
                    module[address] = value
                    self.best_params[address] = value
                else:
                    pass
                
        # Ensure simulators re-calculate with the best parameters
        for simulator in self.simulators:
            simulator()
            simulator._score = self.best_scores

    def _get_best_params_from_study(self):
        """Retrieves the best parameters from the Optuna study.

        This method determines the best trial(s) from the Optuna study and extracts
        their corresponding parameter values.
        - For single-objective optimization (`len(self["directions"]) == 1`), it
          directly uses `self.study.best_params` and `self.study.best_trial`.
        - For multi-objective optimization (`len(self["directions"]) > 1`), it
          iterates through `self.study.best_trials`. Currently, it assumes the
          first objective (`t.values[0]`) is the primary criterion for identifying
          the "best" trial among the Pareto-optimal set.
        Additionally, it merges any extra "best_params" stored in the `user_attrs`
        of the best trial(s). This allows custom callbacks to store additional
        best-fit information.

        Returns
        -------
        dict
            A dictionary of the best parameter names (as they were suggested by Optuna)
            and their corresponding optimized values.
        """
        score_key = str(self["score_function"])
        best_scores = {}

        if len(self["directions"]) > 1:
            best_trials = self.study.best_trials
            best_scores[score_key] = float(
                "inf"
            )   # Initialize with a large value for minimization (assuming first objective is to minimize)
            best_params = {}
            
            for t in best_trials:
                # For multi-objective, we assume the first objective is the primary one for 'best'
                if t.values[0] < best_scores[score_key]:
                    best_scores[score_key] = t.values[0]
                    best_scores["trial_number"] = t.number
                    best_params = t.params
                    if "best_params" in t.user_attrs:
                        best_params = {
                            **best_params,
                            **t.user_attrs["best_params"],
                        }
            return best_params, best_scores
        else:
            best_params = self.study.best_params
            best_trial = self.study.best_trial

            best_scores[score_key] = best_trial.value
            best_scores["trial_number"] = best_trial.number

            if "best_params" in best_trial.user_attrs:
                best_params = {
                    **best_params,
                    **best_trial.user_attrs["best_params"],
                }
            return best_params, best_scores

