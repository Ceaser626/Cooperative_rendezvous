import numpy as np


def equations_of_motion(obs, act, wind_info):
    g = 9.81
    rou = 1.225
    mass = 1.56
    s = 0.2589
    C_D0 = 0.1631
    K = 0.04525
    v_range = [12, 20]
    n_range = [0.95, 1.05]
    gamma_range = [-np.radians(50), np.radians(50)]
    T_range = [0, 2]
    phi_range = [-np.radians(24), np.radians(24)]
    CL_range = [-0.7, 0.7]

    x, y, h, v, eps, gamma = obs
    T, phi, CL = act
    dw_x, dw_y, dw_z = wind_info

    C_D = C_D0 + K * CL**2
    L = 0.5 * rou * v**2 * s * CL
    D = 0.5 * rou * v**2 * s * C_D

    dx_dt = v * np.cos(gamma) * np.cos(eps) + dw_x
    dy_dt = v * np.cos(gamma) * np.sin(eps) + dw_y
    dh_dt = v * np.sin(gamma) + dw_z
    dv_dt = (T - D) / mass - g * np.sin(gamma)
    deps_dt = L * np.sin(phi) / (mass * v * np.cos(gamma))
    dgamma_dt = L * np.cos(phi) / (mass * v) - g * np.cos(gamma) / v

    dX_dt = np.array((dx_dt,
                      dy_dt,
                      dh_dt,
                      dv_dt,
                      deps_dt,
                      dgamma_dt)).squeeze()

    return dX_dt
