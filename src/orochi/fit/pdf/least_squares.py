import numpy as np
import warnings

warnings.simplefilter("ignore")
from ..base.least_squares import LeastSquaresOptimizer
from .pdf_extension import PDFExtensions

class Optimizer(LeastSquaresOptimizer, PDFExtensions):
    r"""
    Performs least-squares optimization for Pair Distribution Function (PDF) calculations using the Lmfit library.

    This class extends `LeastSquaresOptimizer` to specifically handle optimization
    tasks related to Pair Distribution Functions (PDFs) within the Orochi framework.
    It leverages the Lmfit library for robust least-squares fitting.
    While inheriting core optimization functionalities from `LeastSquaresOptimizer`,
    this class provides specific adaptations for PDF data handling and potential
    future extensions related to PDF calculations.


    See Also
    --------
    orochi.base.least_squares.LeastSquaresOptimizer : The base class this optimizer inherits from, providing core Lmfit functionalities.
    lmfit.model.Model : Base class for models in lmfit.
    """

    def __init__(self):
        """
        Initializes the Optimizer class for PDF least-squares optimization.

        This constructor calls the `LeastSquaresOptimizer`'s initializer
        to set up default parameter configurations and core attributes.
        It then specifically sets the `name` and `id` for this PDF optimizer
        instance.
        """

        super().__init__()
        PDFExtensions.__init__(self)

        self._name = "fit.least_squares"
        self._id = "0"

