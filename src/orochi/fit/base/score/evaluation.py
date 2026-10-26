import scipy.interpolate as intp
import numpy as np

from orochi.fit.pdf.score.goodness_of_fit import calc_rw_gof_asd


def calc_gof(obs_x, obs_y):
    def scoring(modules):

        sim_y = np.zeros_like(obs_y)
        for module in modules:
            sim_x0 = module.r
            sim_y0 = module.Gr

            sim_y += intp.interp1d(
                sim_x0, sim_y0, bounds_error=False, fill_value=0.0
            )(obs_x)

        rw, gof, asd = calc_rw_gof_asd(obs_x, obs_y, obs_x, sim_y, weight=1)
        return gof

    return scoring
