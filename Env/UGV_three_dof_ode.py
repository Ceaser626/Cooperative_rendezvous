import numpy as np


def equations_of_motion(obs, act):
    tao_a = 1.5
    tao_eps = 2
    y_range = [-20, 20]
    v_range = [-5, 30]
    a_range = [-5, 5]
    eps_range = [-np.radians(90), np.radians(90)]
    a_c_range = [-5, 5]
    gamma_range = [-np.radians(30), np.radians(30)]

    x, y, v, a, eps = obs
    a_c, eps_c = act

    dx_dt = v * np.cos(eps)
    dy_dt = v * np.sin(eps)
    dv_dt = a
    da_dt = (a_c - a) / tao_a
    deps_dt = (eps_c - eps) / tao_eps

    dX_dt = np.array((dx_dt,
                      dy_dt,
                      dv_dt,
                      da_dt,
                      deps_dt)).squeeze()

    return dX_dt
