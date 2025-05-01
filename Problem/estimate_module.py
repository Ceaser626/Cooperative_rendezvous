import numpy as np
from Env.env_uav import UAV
from Env.env_ugv import UGV
from Utils.tools import interpolate


def estimate_obs(obs_p, obs_s, control_p, control_s, t_k, horizon):

    env_p = UAV()
    env_s = UGV()
    env_p.reset(obs_p)
    env_s.reset(obs_s)

    env_p.t = t_k - horizon
    env_p.dt = min(1, horizon)
    env_s.t = t_k - horizon
    env_s.dt = min(1, horizon)
    f_T = interpolate(control_p.t, control_p.T)
    f_phi = interpolate(control_p.t, control_p.phi)
    f_CL = interpolate(control_p.t, control_p.CL)
    f_a_c = interpolate(control_s.t, control_s.a_c)
    f_eps_c = interpolate(control_s.t, control_s.eps_c)

    for i in range(horizon):
        action_p = [f_T(env_p.t).item(), f_phi(env_p.t).item(), f_CL(env_p.t).item()]
        action_s = [f_a_c(env_s.t).item(), f_eps_c(env_s.t).item()]

        next_obs_p, done, result = env_p.step(action_p)
        next_obs_s, _, result = env_s.step(action_s)
        obs_p = next_obs_p
        obs_s = next_obs_s

    return np.array(obs_p), np.array(obs_s)
