from Problem.uav_control_model import linear_init_p, uav_control_model
from Problem.ugv_control_model import linear_init_s, ugv_control_model
from Utils.solve_ocp import solve
from Env.save_tools import SaveUAV, SaveUGV


def uav_control(t_c, obs_c, theta, ref_traj, nfe_num=10, solve_display_mode=False):

    init_traj = linear_init_p(t_c, obs_c, theta)

    model_c, _ = uav_control_model(t_c, obs_c, theta, ref_traj, init_traj, nfe_num=nfe_num)

    m_c, flag_c = solve(model_c, tee=solve_display_mode)

    ref_traj_c = SaveUAV()
    if flag_c == 'optimal':
        ref_traj_c.store_opt(m_c, store_control=True)
        action = [ref_traj_c.T[1], ref_traj_c.phi[1], ref_traj_c.CL[1]]
    else:
        ref_traj_c.reset()
        action = [0, 0, -0.7]

    return action, ref_traj_c


def ugv_control(t_c, obs_c, theta, ref_traj, nfe_num=10, solve_display_mode=False):

    init_traj = linear_init_s(t_c, obs_c, theta)

    model_c, _ = ugv_control_model(t_c, obs_c, theta, ref_traj, init_traj, nfe_num=nfe_num)

    m_c, flag_c = solve(model_c, tee=solve_display_mode)

    ref_traj_c = SaveUGV()
    if flag_c == 'optimal':
        ref_traj_c.store_opt(m_c, store_control=True)
        action = [ref_traj_c.a_c[1], ref_traj_c.eps_c[1]]
    else:
        ref_traj_c.reset()
        action = [0, 0]

    return action, ref_traj_c
