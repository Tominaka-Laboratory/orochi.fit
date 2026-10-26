import numpy as np
import scipy.interpolate as intp
import copy
import warnings
import json
import os
try:
    import optuna
except ImportError:
    optuna = None

warnings.simplefilter("ignore")
from orochi.base.core import OrochiBase
from .params import common as default  #


class BaseOptimizer(OrochiBase):
    r"""
    Optimizes data using the Lmfit library for various Orochi modules.

    This class serves as a generalized base for performing data optimization
    using the powerful Lmfit library. It integrates seamlessly with the
    "orochi" framework by inheriting from `OrochiBase`, enabling it to
    manage and optimize parameters of one or more "Oroochi" modules
    (e.g., simulators, phases) simultaneously. It constructs a composite
    `lmfit.Model` from these modules to fit observed data.

    Subclasses can inherit from `BaseOptimizer` to implement specific
    optimization workflows (e.g., for PDF data).

    Attributes
    ----------
    models : lmfit.model.CompositeModel or lmfit.model.Model or None
        The `lmfit` model (can be a composite of multiple models) constructed
        from the Orochi modules. It is `None` until `initialize_modules_and_params` is called.
    params : lmfit.Parameters or None
        The `lmfit.Parameters` object containing all optimizable parameters
        for the fit. Created by `set_lmfit_models` from `self.models`. It is `None`
        until `set_lmfit_models` is called.

    Parameters / Keys
    -----------------
    Key configuration parameters accessible via dictionary indexing (e.g., `optimizer["method"]`):
    - "method" : The optimization method for `lmfit` (e.g., "leastsq").
    - "max_nfev" : Maximum number of function evaluations allowed during optimization.
    - "iter_cb" (callable, optional): A callback function to be executed at each optimization iteration.
    - "ftol" (float): Relative tolerance for the change in the sum of squares.
    - "xtol" (float): Relative tolerance for the change in the independent variables.
    - "gtol" (float): Absolute tolerance for the norm of the gradient.
    - "diff_step" (float): Factor for the step size for numerical derivatives when `method` is "least_squares".
    - "obsx" (array_like): The observed independent variable data (x-axis).
    - "obsy" (array_like): The observed dependent variable data (y-axis).
    - "fit_min" (float): The lower bound of the data range to be fitted.
    - "fit_max" (float): The upper bound of the data range to be fitted.
    - "simx_attr" (str): The attribute name on the simulator module that holds the simulated x-axis data (e.g., "r").
    - "simy_attr" (str): The attribute name on the simulator module that holds the simulated y-axis data (e.g., "Gr").

    Attributes (Dynamic / Run-time)
    -------------------------------
    obsx : numpy.ndarray or None
        A subset of the observed independent variable data (`self["obsx"]`)
        that falls within the specified fitting range [`fit_min`, `fit_max`].
        Populated during the `fit` method.
    obsy : numpy.ndarray or None
        A subset of the observed dependent variable data (`self["obsy"]`)
        corresponding to `self.obsx`. Populated during the `fit` method.
    res : lmfit.model.ModelResult or None
        The result object from the `lmfit` optimization process. This contains
        fit statistics, best-fit values, uncertainties, etc. Set by the `fit` method.
    simulators : list of OrochiBase
        A list of Orochi simulator modules that this optimizer will attempt to fit.
        These modules are expected to have a `nest_modules` attribute.
    modules : dict
        A dictionary storing references to all Orochi modules involved in the
        optimization, keyed by their unique `id` (e.g., "0_0", "0_1").
        This includes main simulators and their nested modules.
    best_params : lmfit.Parameters or None
        The best-fit parameters obtained from `self.res.params` after successful
        optimization. Set in `end_process`.
    best_score : float or None
        The primary goodness-of-fit score of the best fit (e.g., R-squared or chi-squared).
        Set by the `fit` method using `score_function`.
    best_scores : dict
        A dictionary storing various goodness-of-fit scores for the best fit.
        Populated in `end_process` or by the `score_function`.
    fit_params : dict
        A dictionary to store detailed information about each parameter to be optimized,
        including its associated module, original address, initial value, min, and max bounds.
        This dictionary serves as the internal representation for building `lmfit.Parameters`.
    score_function : callable or None
        The function used to calculate the goodness-of-fit score. Defaults to a
        goodness-of-fit metric like Rw or user-provided.
    callbacks : list
        A list to store callback functions, primarily for Optuna integration
        (e.g., early stopping, logging).

    Methods
    -------
    report()
        Generates and saves the optimization report, including best parameters
        and scores, to logs and a JSON file. Previously, `report` was a string
        attribute for fit reports, but is now a method for structured reporting.

    See Also
    --------
    orochi.base.core.OrochiBase : The base class for all modules within the Orochi framework.
    lmfit.model.Model : The base class for models used in Lmfit for fitting.
    lmfit.Parameters : A specialized dictionary-like object for managing fit parameters.
    lmfit.model.ModelResult : The object containing the results of an Lmfit model fit.
    """

    def __init__(self):
        """
        Initialize the BaseOptimizer class.

        Sets up default parameter configurations from ``.params.lmfit.default``
        and initializes various attributes essential for the optimization process.
        This includes setting up internal data structures for `lmfit` models,
        parameters, and simulation results.
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

        self.obsx = None
        self.obsy = None
        self.res = None
        self.score_function = None
        self.best_score = None
        self.best_scores = {}

        self.fit_params = {}
        self.init_params = {}
        self.simulators = []
        self.best_params = None

    def __call__(self, force: bool = False):
        """
        Execute the complete optimization process.

        This method orchestrates the entire optimization workflow by sequentially
        calling several internal methods:
        1. Resets internal parameter and model storage to ensure a clean slate.
        2. Calls `initialize_modules_and_params` to configure and build the `lmfit` models from the
           specified Orochi modules.
        3. Calls `set_fit_range` to define the range of observed data to be used for fitting.
        4. Calls `set_score_function` to prepare the function for evaluating goodness-of-fit.
        """
        if force:
            self._state = "INIT"
            if hasattr(self, "simulators") and self.simulators:
                for sim in self.simulators:
                    if hasattr(sim, "_state"):
                        sim._state = "INIT"
        super().__call__(force=force)
        # Reset internal states for a fresh optimization run
        self.fit_params = {}
        self.init_params = {}
        self.initialize_modules_and_params()
        self.set_fit_range()
        self.set_score_function()


    def set_fit_range(self):
        """
        Adjusts simulator's x-axis range and filters observed data for fitting.

        This method iterates through each simulator, extending their `x_min_key`
        and `x_max_key` parameters to ensure they cover the specified `fit_min`
        and `fit_max` range, with an additional `x_extension` for robust
        interpolation. It then filters the `self["obsx"]` and `self["obsy"]`
        arrays to include only data points within the `fit_min` and `fit_max`
        range, storing them in `self.obsx` and `self.obsy`.
        """
        # Resolve raw obsx and obsy from direct setting, _obs_module, or simulators
        raw_x = self["obsx"]
        raw_y = self["obsy"]

        if (raw_x is None or len(raw_x) == 0) and getattr(self, "_obs_module", None) is not None:
            r = getattr(self._obs_module, "r", getattr(self._obs_module, "obs_x", None))
            gr = getattr(self._obs_module, "Gr", getattr(self._obs_module, "obs_y", None))
            if r is not None and gr is not None and len(r) > 0 and len(gr) > 0:
                raw_x = np.asarray(r, dtype=float)
                raw_y = np.asarray(gr, dtype=float)
                self["obsx"] = raw_x
                self["obsy"] = raw_y

        if (raw_x is None or len(raw_x) == 0) and getattr(self, "simulators", None):
            sim0 = self.simulators[0]
            if sim0.obs_x is not None and sim0.obs_y is not None and len(sim0.obs_x) > 0:
                raw_x = np.asarray(sim0.obs_x, dtype=float)
                raw_y = np.asarray(sim0.obs_y, dtype=float)
                self["obsx"] = raw_x
                self["obsy"] = raw_y

        for n, simulator in enumerate(self.simulators):
            ext = max(0.5, float(self["x_extension"])) if self["x_extension"] is not None else 0.5
            if self["fit_min"] is not None:
                xmin = max(0.0001, float(self["fit_min"]) - ext)
                simulator[self["x_min_key"]] = xmin
            if self["fit_max"] is not None:
                xmax = float(self["fit_max"]) + ext
                simulator[self["x_max_key"]] = xmax

        if raw_x is not None and len(raw_x) > 0:
            mask = (raw_x > self["fit_min"]) & (raw_x < self["fit_max"])
            self.obsx = raw_x[mask]
            self.obsy = raw_y[mask] if raw_y is not None else None
        else:
            self.obsx = None
            self.obsy = None

    def set_score_function(self):
        """
        Prepares the observed data and initializes the score function.

        This method filters the `obsx` and `obsy` data based on `fit_min`
        and `fit_max` parameters. It then initializes the `score_function`
        attribute, either using a default goodness-of-fit metric (`calc_rw`)
        or a user-provided callable.
        """

        if self.score_function is None:
            from .score import score_types
            try:
                self.score_function = getattr(score_types, self["score_function"])
            except:
                self.score_function = getattr(score_types, "Rw")

    def initialize_modules_and_params(self):
        """
        Set up `lmfit` models based on the specified Orochi simulator modules.

        This method iterates through each Orochi simulator module provided in
        `self.simulators`. For each simulator:
        1. Assigns a unique `id` to the simulator and its nested modules
           (e.g., "0_0", "0_1"), storing them in `self.modules`.
        2. Adjusts the `xmin` and `xmax` parameters of the simulator to extend
           beyond the fitting range (`self["fit_min"]`, `self["fit_max"]`) by
           `self["x_extension"]`, ensuring sufficient data for interpolation.
        3. Initializes optimizable parameters for both the simulator and its
           `nest_modules` using `_store_current_params`. This populates `self.fit_params`
           (parameter configurations for `lmfit`) and `self.fit_params` (mapping
           `lmfit` names to original Orochi addresses).
        4. Creates an `lmfit.models.Model` instance using `_update_and_get_y`
           as the model function, applying `self.prefix` to parameter names.
        5. If multiple simulator modules are present, they are combined into an
           `lmfit.model.CompositeModel` which is then stored in `self.models`.
           Otherwise, a single `lmfit.Model` is stored.
        """
        self.init_params = {}
        for n, simulator in enumerate(self.simulators):
            simulator._id = f"{n}_0"
            self.init_params[f"{n}_0"] = copy.deepcopy(simulator.par)

            # Initialize parameters for nested modules first
            for m, module_attr in enumerate(simulator["nest_modules"]):
                module = getattr(simulator, module_attr)
                module._id = f"{n}_{m+1}"
                self._store_current_params(module)
                self.init_params[f"{n}_{m+1}"] = copy.deepcopy(module.par)

            # Initialize parameters for the main simulator module
            self._store_current_params(simulator)

        self.clear_changed_params()

    def clear_changed_params(self):
        for simulator in self.simulators:
            simulator._clear_changed_parameters()
            simulator._state = "INIT"

            for m, module_attr in enumerate(simulator["nest_modules"]):
                module = getattr(simulator, module_attr)
                module._clear_changed_parameters()
                module._state = "INIT"
                    
    def _store_current_params(self, module, **kwargs):
        """
        Extract and prepare optimizable parameters from an Orochi module for `lmfit`.

        This method identifies parameters within the given `module` that are
        marked for optimization. It checks `module.optimize_keys` and the
        "optimize" flag for each parameter.

        For each optimizable parameter (whether a single scalar value or an
        element within an array):
        1. An `lmfit`-compatible name (`fit_name`) is constructed. This
           name uses the `self.prefix` (e.g., "s") and replaces characters
           like '/' and '@' in the original Orochi parameter string address
           with underscores (`_`) for `lmfit` compatibility.
        2. This `fit_name` is stored as a key in `self.fit_params`, mapping
           back to the original Orochi string address. This allows reverse
           lookup during parameter propagation.
        3. The parameter's configuration (its current `value`, `min`, `max`,
           and `vary` status) is stored in `self.fit_params` using the
           `fit_name` as the key.
        4. Min/max bounds can be optionally adjusted based on a "tuning_factor"
           provided in `kwargs`. This allows dynamic adjustment of the search
           space around the initial value.

        Parameters
        ----------
        module : OrochiBase
            An instance of an Orochi module (e.g., a Simulator, Phase, or
            any other module with optimizable parameters).
        **kwargs : dict, optional
            Additional keyword arguments that can influence parameter initialization.
            - `tuning_factor` (float, optional): If provided and positive,
              adjusts the `min` and `max` bounds symmetrically around the
              parameter's initial value (`x0`). This can help to narrow
              or widen the search space for optimization.
        """
        
        # Iterate through all keys marked as optimizable/fittable in the module
        for key in module.fit_keys:
            if key not in module.par: continue
            
            x0_val = module[key, "value"]
            opt_flags = module[key, "fit"]

            if np.sum(opt_flags) == 0: continue

            if "opt_func" in module.par[key]:
                self._suggest_func_param(module, key, x0_val)

            # Handle scalar parameters
            elif isinstance(x0_val, (float, int, np.floating, np.integer)):
                self._suggest_numeric_param(module, key, x0_val)

            elif isinstance(x0_val, (str, np.str_)):
                self._suggest_categorical_param(module, key, x0_val)

            # Handle array parameters (e.g., positions)
            elif isinstance(x0_val, (list, np.ndarray)):
                opt_flags_arr = np.array(opt_flags)
                # Find indices where optimization flag is True
                indices = np.where(opt_flags_arr == True)

                for idx_tuple in zip(*indices):
                    idx_tuple = tuple(map(int, idx_tuple))
                    val = x0_val[idx_tuple]
                    if isinstance(val, (float, int, np.floating, np.integer)):
                        self._suggest_numeric_param(module, key, val, idx_tuple)
                    elif "candidates" in module.par[key]:
                        self._suggest_categorical_param(module, key, val, idx_tuple)
                    
            else:
                # Ignore other data types for optimization
                continue

    def _suggest_numeric_param(self, module, key, x0_val=None, idx_tuple=None):
        if idx_tuple is None:
            module_address = module._make_string_address(key, "value")
            if module._check_if_address_constrained(module_address): # Skip if parameter is constrained
                return False

            min_val = module[key, "min"]
            max_val = module[key, "max"]
        else:
            module_address = module._make_string_address(
                key, "value", *idx_tuple
            )
            if module._check_if_address_constrained(module_address): # Skip if constrained
                return False

            min_val = module[module_address.replace("value", "min")]
            max_val = module[module_address.replace("value", "max")]
            x0_val = module[module_address]

        self._store_param(module, module_address, x0_val, min_val, max_val)

    def _suggest_categorical_param(self, module, key, x0_val=None, idx_tuple=None):
        if idx_tuple is None:
            module_address = module._make_string_address(key, "value")
            if module._check_if_address_constrained(module_address): # Skip if parameter is constrained
                return False

            if "candidates" not in module.par[key]:
                return False

            fit_name = module_address.replace(module["address_delimiter"], self["address_delimiter"])
            fit_name = fit_name.replace(module["prefix"], self["prefix"])

            if isinstance(x0_val, (list, tuple, np.ndarray)) and np.size(x0_val) == 1:
                val = x0_val[0]
            else:
                val = x0_val

            self.fit_params[fit_name] = {"module": module, 
                                        "address": module_address,
                                        "fit": True,
                                        "optimize": True,
                                        "value": str(val),
                                        "candidates":module.par[key]["candidates"]}

        else:
            module_address = module._make_string_address(
                key, "value", *idx_tuple
            )
            if module._check_if_address_constrained(module_address): # Skip if constrained
                return False

            candidates = module.par[key]["candidates"][idx_tuple[0]]
            if len(candidates) < 2: return False

            x0_val = module[module_address]
            if isinstance(x0_val, (list, tuple, np.ndarray)) and np.size(x0_val) == 1:
                val = x0_val[0]
            else:
                val = x0_val

            fit_name = module_address.replace(module["address_delimiter"], self["address_delimiter"])
            fit_name = fit_name.replace(module["prefix"], self["prefix"])
            self.fit_params[fit_name] = {"module": module, 
                                         "address": module_address,
                                         "fit": True,
                                         "optimize": True,
                                         "value": str(val),
                                         "candidates":candidates}

    def _suggest_func_param(self, module, key, x0_val=None, idx_tuple=None):

        module_address = module._make_string_address(key, "value")
        if module._check_if_address_constrained(module_address): # Skip if parameter is constrained
            return False

        if "opt_func" not in module.par[key]:
            return False

        fit_name = module_address.replace(module["address_delimiter"], self["address_delimiter"])
        fit_name = fit_name.replace(module["prefix"], self["prefix"])

        if "opt_func_params" in module.par[key]:
            opt_func = getattr(module, module[key, "opt_func"])(**module[key, "opt_func_params"])
        else:
            opt_func = getattr(module, module[key, "opt_func"])()

        self.fit_params[fit_name] = {"module": module, 
                                    "address": module_address,
                                    "fit": True,
                                    "optimize": True,
                                    "value": str(x0_val),
                                    "opt_func":opt_func
                                    }


    def _store_param(self, module, module_address, value, min_val, max_val, **kwargs):
        """
        Helper function to store a single parameter's information into `self.fit_params`.

        This method calculates adjusted `min` and `max` bounds for the parameter
        based on an optional `tuning_factor` provided in `kwargs`. It then
        formats the parameter's details (module, address, initial value,
        adjusted bounds) and adds them to the `self.fit_params` dictionary,
        keyed by an `lmfit`-compatible name derived from the `module_address`.
        Parameters are only stored if their minimum and maximum bounds are not equal,
        indicating that they can vary during optimization.

        Parameters
        ----------
        module : OrochiBase
            The Orochi module to which the parameter belongs.
        module_address : str
            The string address of the parameter within the Orochi module
            (e.g., "@0_0/key/value").
        value : float or int
            The initial value of the parameter.
        min_val : float or int
            The original minimum allowed value for the parameter.
        max_val : float or int
            The original maximum allowed value for the parameter.
        **kwargs : dict, optional
            Additional keyword arguments, potentially containing `tuning_factor`.
        """
        min_adj, max_adj, val_adj = self._get_min_max_with_x0(min_val, max_val, value, kwargs)
        
        # Only add to module_params if min and max are not equal (i.e., parameter can vary)
        if min_adj != max_adj:
            fit_name = module_address.replace(module["address_delimiter"], self["address_delimiter"])
            fit_name = fit_name.replace(module["prefix"], self["prefix"])
            self.fit_params[fit_name] = {"module": module, 
                                         "address": module_address,
                                         "fit": True,
                                         "optimize": True,
                                         "value": val_adj,
                                         "min": min_adj,
                                         "max": max_adj}

    def _get_min_max_with_x0(self, p_min, p_max, p_x0, p_kwargs):
        """
        Helper function to calculate adjusted min/max bounds for a parameter.

        This method takes the original minimum (`p_min`), maximum (`p_max`),
        and initial value (`p_x0`) of a parameter. If a `tuning_factor` is
        provided in `p_kwargs` and is positive, it adjusts the bounds
        symmetrically around `p_x0`. The adjusted bounds are constrained
        to be within the original `p_min` and `p_max`, and `p_x0` is always
        ensured to be within the final calculated bounds.

        Parameters
        ----------
        p_min : float or int
            The original minimum allowed value for the parameter.
        p_max : float or int
            The original maximum allowed value for the parameter.
        p_x0 : float or int
            The initial value of the parameter.
        p_kwargs : dict
            Keyword arguments, potentially containing "tuning_factor".

        Returns
        -------
        tuple
            A tuple containing:
            - `final_min` (float): The adjusted minimum bound.
            - `final_max` (float): The adjusted maximum bound.
            - `p_x0` (float): The initial value (unmodified).
        """
        p_min = float(np.asarray(p_min).flatten()[0])
        p_max = float(np.asarray(p_max).flatten()[0])
        p_x0 = float(np.asarray(p_x0).flatten()[0])
        candidates = np.array([p_min, p_max, p_x0])
        current_min, current_max = np.amin(candidates), np.amax(candidates)

        if "tuning_factor" in p_kwargs:
            tuning_factor = float(p_kwargs["tuning_factor"])
            if tuning_factor > 0:
                original_diff = np.abs(current_max - current_min)
                if original_diff == 0:
                    # If min and max are initially the same, create a small range
                    original_diff = np.abs(p_x0 * 0.1) if p_x0 != 0 else 0.1
                    current_min = p_x0 - original_diff / 2.0
                    current_max = p_x0 + original_diff / 2.0

                half_tuned_range = original_diff / (2.0 * tuning_factor)
                tuned_min = p_x0 - half_tuned_range
                tuned_max = p_x0 + half_tuned_range

                # Ensure tuned range is within original bounds and includes x0
                final_min = np.amax([p_min, tuned_min])
                final_max = np.amin([p_max, tuned_max])

                # If after all adjustments, min is still greater than max, revert to original safe bounds
                if final_min > final_max:
                    final_min, final_max = np.amin([p_min,p_x0]), np.amax([p_max,p_x0])
                return float(final_min), float(final_max), float(p_x0)
            else:
                return float(current_min), float(current_max), float(p_x0)
        return float(current_min), float(current_max), float(p_x0)

    def report(self):
        """
        Generates and saves the optimization report.

        This method compiles a report containing the best-fit parameters and
        best scores obtained from the optimization. It logs this information
        and saves it as a JSON file named "best_params.json" in the
        `self.work_dir/self.study_name` directory.
        """
        report_data = {"best_params": self.best_params, 
                       "best_scores":self.best_scores}     
        report_json = json.dumps(report_data, indent=4)
        report_txt = f"\n optimized by {self._name}\n"+report_json
        self._logger.info(report_txt)

        work_dir = self.work_dir if self.work_dir is not None else "."
        study_name = self.study_name if self.study_name is not None else "default_study"
        dir_name = os.path.join(work_dir, "opt_res", study_name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        file_name = os.path.join(dir_name, "best_params.json")
        with open(file_name, "w") as f:
            f.write(report_json)


    def save_best_result(self):
        """
        Saves the best observed and simulated data from the optimization.

        This method interpolates the simulated y-values from each simulator
        onto the observed x-axis (`self.obsx`). The observed x and y data,
        along with the combined simulated y data, are then saved to a compressed
        NumPy archive (`.npz` file) named "best_result.npz" within the
        `self.work_dir/self.study_name` directory.
        """
        work_dir = self.work_dir if self.work_dir is not None else "."
        study_name = self.study_name if self.study_name is not None else "default_study"
        dir_name = os.path.join(work_dir, "opt_res", study_name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        file_name = os.path.join(dir_name,"best_result.npz")
        x = self.obsx
        simy = []

        for simulator in self.simulators:
            x = getattr(simulator, self["simx_attr"])
            y = getattr(simulator, self["simy_attr"])

            simy_i = intp.interp1d(
                x, y, bounds_error=False, fill_value=0.0
            )(self.obsx)
            simy.append(simy_i)

        if len(self.simulators) == 1:
            simy = simy[0]
        else:
            simy = np.array(simy)
        np.savez(file_name, 
                 obsx=self.obsx, 
                 obsy=self.obsy, 
                 simy=simy,
                 )


    @property
    def simy(self):
        """
        Calculates the best-fit simulated y-values based on the optimized modules.

        This property iterates through the optimized modules, triggers their
        internal calculations (`module()`), and then interpolates their
        simulated data (`module.Gr`) onto the observed x-axis (`self.obsx`).
        The interpolated results from all modules are summed to produce the
        final combined simulated y-values.

        Returns
        -------
        numpy.ndarray
            The simulated y-values corresponding to `self.obsx`, based on the
            best-fit parameters.
        """
        if self.obsx is None or len(self.obsx) == 0:
            return np.array([], dtype=np.float64)

        obsx = np.asarray(self.obsx, dtype=np.float64)
        simy = np.zeros_like(obsx, dtype=np.float64)

        for simulator in self.simulators:
            simulator()
            x = getattr(simulator, self["simx_attr"])
            y = getattr(simulator, self["simy_attr"])
            if x is not None and y is not None and len(x) > 0 and len(y) > 0:
                simy += intp.interp1d(
                    x, y, bounds_error=False, fill_value=0.0
                )(obsx)
        return simy
    
    def set_callbacks(self):
        """
        Sets up Optuna callbacks based on user parameters.

        This method configures the `callbacks` list, adding Optuna callbacks
        such as early stopping and logging, if enabled in the optimizer's parameters.
        The specific callback functions (`_callback_early_stop`, `_callback_logging`)
        are assumed to be defined elsewhere in the class or its superclasses.
        """
        self.callbacks = []
        if self["early_stop"] is not None:
            # Assuming _callback_early_stop is a defined method that returns an Optuna callback
            self.callbacks.append(self._callback_early_stop())
        if self["logging_callback"]:
            # Assuming _callback_logging is a defined method that returns an Optuna callback
            self.callbacks.append(self._callback_logging())
        # Uncomment and implement save_best_params_callback if necessary
        # if self["save_best_params_callback"]:
        #    self.callbacks.append(self._callback_save_best_params())

    def set_obs_data(self, obsx, obsy=None):
        """
        Set the observed data for the optimizer.
        Supports passing:
        - An Orochi module producing r and Gr (e.g. `PDFget` instance),
        - A Simulator instance with `obs_x` and `obs_y`, OR
        - Raw arrays (obsx, obsy).
        """
        # If passed an Orochi module directly (e.g., pdfget)
        if hasattr(obsx, "r") and (hasattr(obsx, "Gr") or hasattr(obsx, "gr")):
            self._obs_module = obsx
            if getattr(self, "simulators", None):
                for sim in self.simulators:
                    sim.set_obs_data(obsx)
            r = getattr(obsx, "r", None)
            gr = getattr(obsx, "Gr", None)
            if gr is None and not hasattr(obsx, "Gr") and hasattr(obsx, "gr"):
                gr = getattr(obsx, "gr", None)
            if r is not None and gr is not None and len(r) > 0 and len(gr) > 0:
                self["obsx"] = np.asarray(r, dtype=float)
                self["obsy"] = np.asarray(gr, dtype=float)
                try:
                    self.set_fit_range()
                except Exception:
                    pass
            return self

        # If passed a Simulator directly
        if hasattr(obsx, "obs_x") and hasattr(obsx, "obs_y"):
            self._obs_module = obsx
            ox = obsx.obs_x
            oy = obsx.obs_y
            if ox is not None and oy is not None and len(ox) > 0 and len(oy) > 0:
                self["obsx"] = np.asarray(ox, dtype=float)
                self["obsy"] = np.asarray(oy, dtype=float)
                try:
                    self.set_fit_range()
                except Exception:
                    pass
            return self

        self["obsx"] = obsx
        self["obsy"] = obsy
        if self.simulators:
            try:
                self.set_fit_range()
            except Exception:
                self.obsx = obsx
                self.obsy = obsy
        else:
            self.obsx = obsx
            self.obsy = obsy
        return self

    @property
    def obs_x(self):
        """Observed x grid (filtered by fit range if set, else full obsx)."""
        if getattr(self, "obsx", None) is not None:
            return self.obsx
        if self["obsx"] is not None:
            return self["obsx"]
        if getattr(self, "_obs_module", None) is not None:
            return getattr(self._obs_module, "r", getattr(self._obs_module, "obs_x", None))
        return None

    @property
    def obs_y(self):
        """Observed y values (filtered by fit range if set, else full obsy)."""
        if getattr(self, "obsy", None) is not None:
            return self.obsy
        if self["obsy"] is not None:
            return self["obsy"]
        if getattr(self, "_obs_module", None) is not None:
            return getattr(self._obs_module, "Gr", getattr(self._obs_module, "obs_y", None))
        return None

    # utilities
    def use_as_score_function(self, conditions, nest_study_name="sub_trial"):
        """create a nest optimization"""

        def score_function(obsx, obsy, simx_attr, simy_attr, **score_kwargs):

            def scoring(simulators):
                if "trial" in score_kwargs:
                    self.study_name = f"{nest_study_name}_{score_kwargs['trial'].number}"
                optuna.logging.set_verbosity(optuna.logging.WARNING)
                self.run_with_list_conditions(conditions)
                optuna.logging.set_verbosity(optuna.logging.INFO)

                return self.best_scores[str(self["score_function"])]
            return scoring
        return score_function

    def run_with_list_conditions(self, conditions:list):
        ori_mod_par = {}
        for con in conditions:
            mod  = con.get("module", None)
            if id(mod) not in ori_mod_par:
                ori_mod_par[id(mod)] = {
                    "par": copy.deepcopy(mod.par),
                    "mod": mod,
                }
                for k, v in mod.par.items():
                    if isinstance(v, dict):
                        if "fit" in v:
                            mod[k, "fit"] = np.zeros_like(v["fit"], dtype=bool) if hasattr(v["fit"], "__iter__") else False
                        elif "optimize" in v:
                            mod[k, "optimize"] = np.zeros_like(v["optimize"], dtype=bool) if hasattr(v["optimize"], "__iter__") else False

        self._set_refine_conditions(conditions)
        self()

        for mod_id, info in ori_mod_par.items():
            mod = info["mod"]
            mod.par = copy.deepcopy(info["par"])
            mod._clear_changed_parameters()

    def _set_refine_conditions(self, conditions: list):

        ordered_conds = {}

        for con in conditions:
            order = con.get("order", 1)
            if order not in ordered_conds:
                ordered_conds[order] = []
            ordered_conds[order].append(con)

        # Execute in ascending order (0 -> 1 -> 2...)
        for order in sorted(ordered_conds.keys()):
            cond_list = ordered_conds[order]
        
            for con in cond_list:
                if "func" in con and callable(con["func"]): 
                    con["func"]()
                elif "forward" in con and callable(con["forward"]): 
                    con["forward"]() 

                mod = con.get("module", None)
                key = str(con.get("key", None))

                if mod is not None and key is not None:
                    attr = con.get("attr", "value")

                    val0 = con.get("val", None)
                    if callable(val0):
                        val = val0()
                    elif val0 is None and attr in ("optimize", "fit"):
                        current_val = mod[key]
                        val = np.ones_like(current_val).astype(bool)
                    else:
                        val = val0
                        
                    if attr in ("optimize", "fit"):
                        mod[key, "fit"] = val
                    else:
                        mod[key, attr] = val

       
    def _reset_refine_conditions(self, conditions: list):

        ordered_conds = {}

        for con in conditions:
            order = con.get("order", 1)
            if order not in ordered_conds:
                ordered_conds[order] = []
            ordered_conds[order].append(con)

        # Execute in descending order (...2 -> 1 -> 0)
        for order in sorted(ordered_conds.keys(), reverse=True):
            cond_list = ordered_conds[order]

            for con in cond_list:

                if "backward" in con and callable(con["backward"]): 
                    con["backward"]() 

                # 2. Reset parameters afterwards (order = 0 runs last)
                mod = con.get("module", None)
                key = con.get("key", None)
                attr = con.get("attr", "value")

                if mod is not None and key is not None:
                    current_val = mod[key]

                    if current_val is not None and np.ndim(current_val) != 0:
                        indices = con.get("indices", None)
                        n_pts = np.shape(current_val)[0]

                        if indices is None:
                            indices = list(range(n_pts))
                        elif isinstance(indices, int):
                            indices = [indices]
                        elif isinstance(indices, np.ndarray):
                            indices = indices.astype(int).tolist()
                        elif isinstance(indices, list):
                            indices = [int(i) for i in indices]
                        else:
                            indices = list(range(n_pts))
                        
                        indices = [int(i) for i in indices if i < n_pts]
                    else:
                        indices = None
                        
                    if attr in ("optimize", "fit"):
                        if indices is None:
                            mod[key, "fit"] = False
                        else:
                            opt_bools = np.zeros_like(current_val, dtype=bool)            
                            if opt_bools.size == 1:
                                opt_bools = False
                            else:         
                                opt_bools[indices] = False
                            mod[key, "fit"] = opt_bools

    def read_file(self, file_path: str):
        """
        Reads observed Pair Distribution Function (PDF) or 1D scattering data from a specified file.

        This method loads data, extracts the independent variable (`obsx` as r or 2theta/Q)
        and dependent variable (`obsy` as G(r) or I(Q)), and assigns them to `self["obsx"]`
        and `self["obsy"]`. Additionally, it attempts to extract metadata from file headers
        (e.g., 'rmin') and applies relevant values to the attached simulator modules.

        Parameters
        ----------
        file_path : str
            The path to the experimental data file.
        """
        try:
            from orochi.utils.file import read_data
            data, meta = read_data.Gr(file_path)
        except (ImportError, ModuleNotFoundError):
            meta = {}
            pts = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s.startswith("#"):
                        if "=" in s:
                            k, v = s.lstrip("#").split("=", 1)
                            meta[k.strip().lower()] = v.strip()
                    elif s:
                        parts = s.split()
                        if len(parts) >= 2:
                            try:
                                pts.append([float(parts[0]), float(parts[1])])
                            except ValueError:
                                pass
            data = np.array(pts)

        obsx = data[:, 0]
        obsy = data[:, 1]
        self["obsx"] = obsx
        self["obsy"] = obsy

        for key, value in meta.items():
            for sim in self.simulators:
                if key in sim.par:
                    if key == "rmin":
                        if float(value) <= 0.01:
                            value = 0.01
                    sim[key] = value

    def save_file(self, file_path: str):
        """
        Save fitted experimental and simulated data to a specified CSV file.

        Parameters
        ----------
        file_path : str
            The output path for the saved data file.
        """
        import pandas as pd
        import os

        res = {}
        res["obsx"] = self.obsx
        res["obsy"] = self.obsy

        for simulator in self.simulators:
            x = getattr(simulator, self["simx_attr"])
            y = getattr(simulator, self["simy_attr"])
            simy_i = intp.interp1d(
                x, y, bounds_error=False, fill_value=0.0
            )(self.obsx)
            res[simulator._name] = simy_i

        data = pd.DataFrame(res)

        ext = os.path.splitext(file_path)[1]
        if len(ext) == 0:
            file_path += ".csv"

        data.to_csv(file_path, index=False)

    @property
    def Rw(self) -> float:
        """Returns the weighted profile agreement factor Rw for the current fit."""
        if hasattr(self, "best_scores") and isinstance(self.best_scores, dict):
            if "Rw" in self.best_scores and self.best_scores["Rw"] is not None:
                return float(self.best_scores["Rw"])
        if hasattr(self, "obsy") and self.obsy is not None and len(self.obsy) > 0:
            denom = float(np.sum(np.square(self.obsy)))
            if denom > 0.0:
                return float(np.sqrt(np.sum(np.square(self.obsy - self.simy)) / denom))
        return None

