import numpy as np
import warnings

warnings.simplefilter("ignore")
from ..base.bayes import BayesOptimizer
from .pdf_extension import PDFExtensions


class Optimizer(BayesOptimizer, PDFExtensions):
    """Performs Bayesian optimization for Pair Distribution Function (PDF) calculations using the Optuna library.

    This class extends `BayesOptimizer` to specifically handle optimization
    tasks related to Pair Distribution Functions (PDFs) within the Orochi framework.
    It leverages the `Optuna` library for Bayesian optimization, allowing for
    efficient exploration of parameter spaces for PDF simulations.
    While inheriting core Bayesian optimization functionalities from `BayesOptimizer`,
    this class provides specific adaptations for PDF data handling, notably
    through its `read_file` method.


    See Also
    --------
    orochi.base.bayes.BayesOptimizer : The base class this optimizer inherits from, providing core Optuna functionalities.
    optuna : The Optuna library for hyperparameter optimization.
    """

    def __init__(self):
        """Initializes the Optimizer class for PDF Bayesian Optimization using Optuna.

        This constructor calls the `BayesOptimizer`'s initializer to set up
        default parameter configurations and core attributes. It then specifically
        sets the `name` and `id` for this PDF optimizer instance.
        """

        super().__init__()
        # optuna.logging.disable_default_handler() # Uncomment if Optuna logs are too verbose
        self._name = "fit.bayes"
        self._id = "3"  # Unique identifier for this optimizer module
