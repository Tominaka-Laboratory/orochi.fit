.. image:: images/orochi_icon.png
   :align: center
   :width: 80px
   :target: https://Tominaka-Laboratory.github.io/orochi.fit
   :alt: Orochi Logo

==========
orochi.fit
==========

.. |title| replace:: orochi.fit
.. _title: https://Tominaka-Laboratory.github.io/orochi.fit

.. |Icon| image:: images/orochi_icon.png
   :target: https://Tominaka-Laboratory.github.io/orochi.fit
   :height: 100px

|PyPi| |Forge| |PythonVersion| |PR|

|CI| |Codecov| |Black| |Tracking|

.. |Black| image:: https://img.shields.io/badge/code_style-black-black
        :target: https://github.com/psf/black

.. |CI| image:: https://github.com/Tominaka-Laboratory/orochi.fit/actions/workflows/tests.yml/badge.svg
        :target: https://github.com/Tominaka-Laboratory/orochi.fit/actions/workflows/tests.yml

.. |Codecov| image:: https://codecov.io/gh/Tominaka-Laboratory/orochi.fit/branch/main/graph/badge.svg
        :target: https://codecov.io/gh/Tominaka-Laboratory/orochi.fit

.. |Forge| image:: https://img.shields.io/conda/vn/conda-forge/orochi.fit
        :target: https://anaconda.org/conda-forge/orochi.fit

.. |PR| image:: https://img.shields.io/badge/PR-Welcome-29ab47ff

.. |PyPi| image:: https://img.shields.io/pypi/v/orochi.fit
        :target: https://pypi.org/project/orochi.fit/

.. |PythonVersion| image:: https://img.shields.io/pypi/pyversions/orochi.fit
        :target: https://pypi.org/project/orochi.fit/

.. |Tracking| image:: https://img.shields.io/badge/issue_tracking-github-blue
        :target: https://github.com/Tominaka-Laboratory/orochi.fit/issues

---
Orochi Fit: Parameter Refinement and Multi-Module Optimization
---

The ``orochi.fit`` package provides robust structural refinement and parameter optimization engines for experimental scattering data within the **Orochi 4.0** ecosystem. It bridges physical simulators (such as ``orochi.pdfsim`` and ``orochi.phase``) with modern numerical optimization frameworks:

* **Bounded Least-Squares Refinement (``LeastSquaresOptimizer``)**: Powered by `lmfit <https://lmfit.github.io/lmfit-py/>`_ and SciPy, supporting Levenberg--Marquardt, Nelder--Mead, and Powell algorithms with parameter bounds, linear constraints, and parameter ties.
* **Bayesian Global Optimization (``BayesOptimizer``)**: Powered by `Optuna <https://optuna.org/>`_, enabling tree-structured Parzen estimator (TPE) and CMA-ES sampling for challenging, multi-modal, or highly correlated parameter landscapes.

Quickstart: Real-Space PDF Refinement
-------------------------------------

The following example demonstrates refining the scale factor and atomic displacement parameter (*U_iso*) of a ceria (CeO₂) model against experimental Pair Distribution Function (*G(r)*) data:

.. code-block:: python

    import numpy as np
    from orochi.phase import Phase
    from orochi.pdfsim import Simulator
    from orochi.fit import LeastSquaresOptimizer

    # 1. Setup Phase and Simulator
    phase = Phase()
    phase.read_cif("CeO2.cif")

    sim = Simulator()
    sim.phase = phase
    sim["qmax"] = 25.0
    sim["qdamp"] = 0.04

    # 2. Configure optimizable parameters and search bounds
    sim["scale"] = 1.0
    sim["scale", "min"] = 0.1
    sim["scale", "max"] = 5.0
    sim["scale", "fit"] = True

    phase["uiso_0"] = 0.005
    phase["uiso_0", "min"] = 0.0001
    phase["uiso_0", "max"] = 0.05
    phase["uiso_0", "fit"] = True

    # 3. Configure Optimizer
    opt = LeastSquaresOptimizer()
    opt.simulators = [sim]
    opt["fit_min"] = 1.5
    opt["fit_max"] = 20.0
    opt.study_name = "ceo2_refinement"

    # Load experimental PDF data (r, G(r))
    opt.read_file("CeO2_experimental.gr")

    # 4. Execute Refinement
    # Automated execution:
    opt()

    # Or explicit manual execution forcing clean re-evaluation:
    # opt.run(force=True)

    # 5. Inspect Results
    print("Optimization Scores:", opt.best_scores)
    print("Refined Parameters:", opt.best_params)

Execution Lifecycle & Force Recalculation
-----------------------------------------

Like all core Orochi modules, ``LeastSquaresOptimizer`` and ``BayesOptimizer`` support dual execution interfaces:

* **Automated execution via ``opt()``**: Resets internal states and optimizes according to changed parameters.
* **Manual execution via ``opt.run(force=True)``**: Forcibly resets both the optimizer's cache and all attached simulators' caches (setting ``_state = "INIT"``), guaranteeing a completely fresh evaluation from initial conditions.

Installation
------------

Install via ``pip``:

.. code-block:: bash

    pip install orochi.fit

Or install in development mode from source:

.. code-block:: bash

    git clone https://github.com/Tominaka-Laboratory/orochi.fit.git
    cd orochi.fit
    pip install -e .

To confirm successful installation:

.. code-block:: bash

    python -c "import orochi.fit; print(orochi.fit.__version__)"

Citation
--------

If you use ``orochi.fit`` in your research, please cite:

    Orochi Software Suite, Tominaka Laboratory, National Institute for Materials Science (NIMS).
    https://github.com/Tominaka-Laboratory/orochi.fit

For Bayesian optimization workflows, please also cite Optuna:

    Akiba et al., "Optuna: A Next-generation Hyperparameter Optimization Framework",
    *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (2019).



License
-------

This project is licensed under the **GNU General Public License v3.0 (GPLv3)**. See the `LICENSE.rst` file for details.

This copyleft license guarantees that Orochi remains freely accessible, open, and transparent to the entire scientific community, and that all derivative works must likewise remain open source.

Citation
--------

If you use this package in academic research, please cite:

.. code-block:: bibtex

    @article{tominaka2026orochi,
      title   = {Orochi: A Physics-Grounded Python Package for Pair Distribution Function Analysis of Crystalline and Low-Dimensional Materials},
      author  = {Tominaka, Satoshi},
      year    = {2026},
      url     = {https://github.com/Tominaka-Laboratory/orochi}
    }

Contact
-------

For bug reports and questions, visit `GitHub Issues <https://github.com/Tominaka-Laboratory/orochi.fit/issues>`_ or contact Dr. Satoshi Tominaka at ``tominaka.satoshi@nims.go.jp``.
