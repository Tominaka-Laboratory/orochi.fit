import numpy as np
import scipy.interpolate as intp

bolzman = 1.38065 * 10**-23  # J/K
epsilon = 120 * bolzman  # J; assuming a constant value

def Rw(obsx, obsy, simx_attr, simy_attr, weight=1.0, **kwargs):
    """
    Calculate the standard weighted R-factor (Rw) for PDF data fitting.

    This function quantifies the normalized goodness-of-fit between experimental 
    and simulated pair distribution function (PDF) profiles.

    Args:
        obsx (ndarray): Experimental x-axis array (r values in Å).
        obsy (ndarray): Experimental y-axis array (G(r) values).
        simx_attr (str): Attribute name for simulated x-axis in the module (e.g., "x").
        simy_attr (str): Attribute name for simulated y-axis in the module (e.g., "y").
        weight (float or ndarray, optional): Statistical weight array or scalar factor for fitting. Defaults to 1.0.
        **kwargs: Additional keyword arguments for compatibility.

    Returns:
        function: A closure `score(modules)` that takes a list of simulator modules 
                  and returns the scalar Rw value (dimensionless, typically 0.0 to 1.0).

    Notes / Strategy Tips for LLM:
        - **Primary Application**: Recommended for macro-parameter refinement (lattice constants, 
          scale, particle size) and heavy-atom framework exploration (e.g., Ru sublattice).
        - **Limitation**: Pure Rw can be insensitive to light-element coordinates (e.g., Oxygen) 
          under strong Z-contrast, potentially driving light atoms into unphysical short contacts 
          to achieve minor Rw gains. Use `rw_lj_penalty` for light-element relaxation.

    Example:
        >>> score_fn = Rw(obsx, obsy, simx_attr="x", simy_attr="y")
        >>> current_rw = score_fn(simulator.modules)
    """

    def score(modules):
        simy = np.zeros_like(obsx)
        for module in modules:
            x = getattr(module, simx_attr)
            y = getattr(module, simy_attr)

            simy += intp.interp1d(
                x, y, bounds_error=False, fill_value=0.0
            )(obsx)

        rw = np.sqrt(
            np.average(weight * np.square(obsy - simy))
            / np.average(weight * np.square(obsy))
        )
        return rw

    return score


def LJ_penalty(obsx, obsy, simx_attr, simy_attr, weight=1.0, **kwargs):
    """
    Compute a multiplicative geometric penalty factor (P_factor >= 1.0) based on Lennard-Jones repulsion.

    This function acts as a physical validity filter. It evaluates site-wise potential energies 
    using periodic boundary conditions (PBC) and Lennard-Jones (12-6) interactions.
    Negative energy wells (attractive force) are clamped to 0.0, isolating only unphysical 
    atomic overlaps/repulsions (total_overlap > overlap_threshold) to scale up the penalty factor.

    Args:
        obsx (ndarray): Experimental x-axis array (used for interface compatibility).
        obsy (ndarray): Experimental y-axis array (used for interface compatibility).
        simx_attr (str): Attribute name for simulated x-axis.
        simy_attr (str): Attribute name for simulated y-axis.
        weight (float, optional): Weight parameter (unused, maintained for API consistency).
        **kwargs:
            - overlap_threshold (float, optional): Repulsion energy offset. Defaults to 0.0.
            - penalty_weight (float, optional): Scaling factor for the logarithmic penalty. Defaults to 1.0.

    Returns:
        function: A closure `score(modules)` that takes a list of simulator modules 
                  and returns the dimensionless penalty factor `P_factor` (P_factor = 1.0 for valid 
                  structures, P_factor > 1.0 when atomic overlaps occur).

    Notes / Strategy Tips for LLM:
        - **Behavior**: 
          * Physically reasonable distances (e.g., O-O >= 2.2 Å) -> `P_factor = 1.0`
          * Unphysical atomic overlaps (e.g., O-O < 2.2 Å) -> `P_factor = 1.0 + log1p(repulsion)` (> 1.0)
        - **Usage**: Typically combined with `Rw` via `rw_lj_penalty` rather than used alone.

    Example:
        >>> penalty_fn = LJ_penalty(obsx, obsy, simx_attr="x", simy_attr="y", overlap_threshold=0.0)
        >>> p_factor = penalty_fn(simulator.modules)
    """

    threshold = kwargs.get("threshold", -0.05)
    penalty_weight = kwargs.get("penalty_weight", 1.0)

    def score(modules):
        P_factor = 1.
        for module in modules:
            if "nest_modules" in module.par:
                for mod_name in module["nest_modules"]:
                    if "phase" in mod_name:
                        phase = getattr(module, "phase")

                        include_coulomb = kwargs.get("include_coulomb", False)
                        scores = phase.eval_site_scores(polyhedral=False, overlap=True, nearest=False, include_coulomb=include_coulomb)
                        site_energies = scores['overlap'] # Potential energy for each site

                        penalty = np.maximum(0.0, site_energies/epsilon - threshold)
                        P_factor *= 1.0 + penalty_weight * np.log1p(np.sum(penalty) )
                        
        return P_factor
    return score


def window_penalty(obsx, obsy, simx_attr, simy_attr, weight=1.0, **kwargs):
    """
    Compute a harmonic window-based geometric penalty factor based on nearest-neighbor distance ratios.

    This function enforces a physical bond-length window [r_repel, r_attract] using normalized 
    nearest-neighbor distance ratios (scores["nearest"]). It applies a quadratic (harmonic) 
    penalty when atomic distances are either too short (unphysical overlap/repulsion) or 
    too long (isolated/unbound atoms).

    Args:
        obsx (ndarray): Experimental x-axis array (used for interface compatibility).
        obsy (ndarray): Experimental y-axis array (used for interface compatibility).
        simx_attr (str): Attribute name for simulated x-axis.
        simy_attr (str): Attribute name for simulated y-axis.
        weight (float, optional): Weight parameter (maintained for API consistency).
        **kwargs:
            - k_repel (float, optional): Harmonic penalty coefficient for short-distance repulsion (ratio < r_repel). Defaults to 100.0.
            - k_attract (float, optional): Harmonic penalty coefficient for long-distance isolation (ratio > r_attract). Defaults to 50.0.
            - r_repel (float, optional): Lower ratio threshold for repulsion penalty. Defaults to 0.7.
            - r_attract (float, optional): Upper ratio threshold for isolation penalty. Defaults to 1.5.
            - include_coulomb (bool, optional): Whether to include Coulomb evaluation. Defaults to False.

    Returns:
        function: A closure `score(modules)` that takes a list of simulator modules 
                  and returns the calculated geometric penalty factor `P_factor`.

    Notes / Strategy Tips for LLM:
        - **Mechanism**: 
          * Evaluates nearest-neighbor distance ratios (`ratio = scores["nearest"]`).
          * Repulsion Penalty: `k_repel * (r_repel - ratio)^2` when `ratio < r_repel`.
          * Isolation Penalty: `k_attract * (ratio - r_attract)^2` when `ratio > r_attract`.
          * Valid Window `[r_repel, r_attract]`: No penalty applied.
        - **Usage**: Bounds bond lengths within a valid physical window to prevent 
          both atomic collapse and atom detachment during coordinate refinement.

    Example:
        >>> penalty_fn = window_penalty(obsx, obsy, simx_attr="x", simy_attr="y", r_repel=0.7, r_attract=1.5)
        >>> p_factor = penalty_fn(simulator.modules)
    """

    k_repel = kwargs.get("k_repel", 100.0)
    k_attract = kwargs.get("k_attract", 50.0)
    r_repel = kwargs.get("r_repel", 0.7)
    r_attract = kwargs.get("r_attract", 1.5)

    def score(modules):
        P_factor = 1.
        for module in modules:
            if "nest_modules" in module.par:
                for mod_name in module["nest_modules"]:
                    if "phase" in mod_name:
                        phase = getattr(module, "phase")
                        scores = phase.eval_site_scores(polyhedral=False, 
                                                        overlap=False, 
                                                        nearest=True, 
                                                        include_coulomb=False)

                        ratio = scores["nearest"]
                       
                        # --- 1. Short-distance repulsion penalty (ratio < r_repel) ---
                        overlap_mask = ratio < r_repel
                        penalty_repel = np.sum(k_repel * (r_repel - ratio[overlap_mask])**2)
                        
                        # --- 2. Long-distance isolation penalty (ratio > r_attract) ---
                        isolation_mask = ratio > r_attract
                        penalty_attract = np.sum(k_attract * (ratio[isolation_mask] - r_attract)**2)
                        
                        total_penalty = penalty_repel + penalty_attract
                        P_factor *= (1.0 + total_penalty)
                        
        return P_factor
    return score


def rw_lj_penalty(obsx, obsy, simx_attr, simy_attr, weight=1.0, **kwargs):
    """
    Calculate the combined objective score (Rw * P_factor) for constrained light-element refinement.

    This function multiplies the experimental PDF fitting error (Rw) by the physical collision 
    penalty factor (P_factor). It prevents optimizers (e.g., Optuna CMA-ES) from driving light 
    atoms (e.g., Oxygen) into unphysical local minima (bond collapse) during coordinate optimization.

    Args:
        obsx (ndarray): Experimental x-axis array (r values in Å).
        obsy (ndarray): Experimental y-axis array (G(r) values).
        simx_attr (str): Attribute name for simulated x-axis.
        simy_attr (str): Attribute name for simulated y-axis.
        weight (float or ndarray, optional): Statistical fitting weight. Defaults to 1.0.
        **kwargs: Arguments passed to both `Rw` and `LJ_penalty` (e.g., `overlap_threshold`).

    Returns:
        function: A closure `score(modules)` returning `rw * P_factor` (float).

    Notes / Strategy Tips for LLM:
        - **RECOMMENDED STAGE**: Mandatory for Stage 2 refinement (Oxygen / light element relaxation).
        - **Mathematical Advantage**: Multiplicative form ensures scale-invariance. In physically 
          valid regions (`P_factor = 1.0`), the returned value is exactly equal to pure `Rw`. When 
          atomic collisions occur, the score scales up dramatically, creating an effective barrier 
          against unphysical coordination geometries.

    Example:
        >>> combined_score_fn = rw_lj_penalty(obsx, obsy, simx_attr="x", simy_attr="y")
        >>> total_loss = combined_score_fn(simulator.modules)
    """

    def score(modules):
        rw = Rw(obsx, obsy, simx_attr, simy_attr, weight=1.0, **kwargs)(modules)
        P_factor = LJ_penalty(obsx, obsy, simx_attr, simy_attr, weight=1.0, **kwargs)(modules)
        return rw * P_factor
    return score

def rw_ljc_penalty(obsx, obsy, simx_attr, simy_attr, weight=1.0, **kwargs):
    """
    Calculate the combined objective score (Rw * P_factor) for constrained light-element refinement.

    This function multiplies the experimental PDF fitting error (Rw) by the physical collision 
    penalty factor (P_factor). It prevents optimizers (e.g., Optuna CMA-ES) from driving light 
    atoms (e.g., Oxygen) into unphysical local minima (bond collapse) during coordinate optimization.

    Args:
        obsx (ndarray): Experimental x-axis array (r values in Å).
        obsy (ndarray): Experimental y-axis array (G(r) values).
        simx_attr (str): Attribute name for simulated x-axis.
        simy_attr (str): Attribute name for simulated y-axis.
        weight (float or ndarray, optional): Statistical fitting weight. Defaults to 1.0.
        **kwargs: Arguments passed to both `Rw` and `LJ_penalty` (e.g., `overlap_threshold`).

    Returns:
        function: A closure `score(modules)` returning `rw * P_factor` (float).

    Notes / Strategy Tips for LLM:
        - **RECOMMENDED STAGE**: Mandatory for Stage 2 refinement (Oxygen / light element relaxation).
        - **Mathematical Advantage**: Multiplicative form ensures scale-invariance. In physically 
          valid regions (`P_factor = 1.0`), the returned value is exactly equal to pure `Rw`. When 
          atomic collisions occur, the score scales up dramatically, creating an effective barrier 
          against unphysical coordination geometries.

    Example:
        >>> combined_score_fn = rw_lj_penalty(obsx, obsy, simx_attr="x", simy_attr="y")
        >>> total_loss = combined_score_fn(simulator.modules)
    """

    def score(modules):
        rw = Rw(obsx, obsy, simx_attr, simy_attr, weight=1.0, **kwargs)(modules)
        P_factor = LJ_penalty(obsx, obsy, simx_attr, simy_attr, weight=1.0, include_coulomb=True, **kwargs)(modules)
        return rw * P_factor
    return score

def rw_window_penalty(obsx, obsy, simx_attr, simy_attr, weight=1.0, **kwargs):
    """
    Calculate the combined objective score (Rw * P_factor) for constrained light-element refinement.

    This function multiplies the experimental PDF fitting error (Rw) by the physical collision 
    penalty factor (P_factor). It prevents optimizers (e.g., Optuna CMA-ES) from driving light 
    atoms (e.g., Oxygen) into unphysical local minima (bond collapse) during coordinate optimization.

    Args:
        obsx (ndarray): Experimental x-axis array (r values in Å).
        obsy (ndarray): Experimental y-axis array (G(r) values).
        simx_attr (str): Attribute name for simulated x-axis.
        simy_attr (str): Attribute name for simulated y-axis.
        weight (float or ndarray, optional): Statistical fitting weight. Defaults to 1.0.
        **kwargs: Arguments passed to both `Rw` and `LJ_penalty` (e.g., `overlap_threshold`).

    Returns:
        function: A closure `score(modules)` returning `rw * P_factor` (float).

    Notes / Strategy Tips for LLM:
        - **RECOMMENDED STAGE**: Mandatory for Stage 2 refinement (Oxygen / light element relaxation).
        - **Mathematical Advantage**: Multiplicative form ensures scale-invariance. In physically 
          valid regions (`P_factor = 1.0`), the returned value is exactly equal to pure `Rw`. When 
          atomic collisions occur, the score scales up dramatically, creating an effective barrier 
          against unphysical coordination geometries.

    Example:
        >>> combined_score_fn = rw_lj_penalty(obsx, obsy, simx_attr="x", simy_attr="y")
        >>> total_loss = combined_score_fn(simulator.modules)
    """

    def score(modules):
        rw = Rw(obsx, obsy, simx_attr, simy_attr, weight=1.0, **kwargs)(modules)
        P_factor = window_penalty(obsx, obsy, simx_attr, simy_attr, weight=1.0, **kwargs)(modules)
        return rw * P_factor
    return score





def GOF(obsx, obsy, simx_attr, simy_attr, weight=1.0, **kwargs):
    def score(modules):
        simy = np.zeros_like(obsx)
        for module in modules:
            x = getattr(module, simx_attr)
            y = getattr(module, simy_attr)

            simy += intp.interp1d(
                x, y, bounds_error=False, fill_value=0.0
            )(obsx)

        gof = np.sqrt(
            np.average(weight * np.square(obsy - simy))
        )  # / (np.size(obs_y) - number_of_variables))
        return gof

    return score


def ASD(obsx, obsy, simx_attr, simy_attr, weight=1.0, **kwargs):
    def score(modules):
        simy = np.zeros_like(obsx)
        for module in modules:
            x = getattr(module, simx_attr)
            y = getattr(module, simy_attr)

            simy += intp.interp1d(
                x, y, bounds_error=False, fill_value=0.0
            )(obsx)

        sigma_obs_sim = np.average(weight * np.square(obsy - simy))
        asd = sigma_obs_sim / np.size(obsy)

        return asd

    return score
