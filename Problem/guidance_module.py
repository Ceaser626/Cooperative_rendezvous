import numpy as np
from pyomo.environ import value
from pyomo.contrib.sensitivity_toolbox.sens import SensitivityInterface, SolverFactory

from Problem.uav_ocp_model import decentralized_uav, linear_init_uav
from Problem.ugv_ocp_model import decentralized_ugv, linear_init_ugv
from Utils.solve_ocp import solve, sipopt_solve
from Env.save_tools import SaveUAV
from Env.save_tools import SaveUGV


def vehicle_opt(t, obs, theta, TR, ref, vehicle_num, nfe_num=10, sens=True, wind=None, estimator=None):
    p, g, g_prev, B, iter_num = TR.p, TR.g, TR.g_prev, TR.B, TR.iter_num
    scale = 1

    if vehicle_num == 1:
        m, V, traj = uav_solve(t, obs, theta, ref, nfe_num=nfe_num)
    else:
        m, V, traj = ugv_solve(t, obs, theta, ref, nfe_num=nfe_num)

    dx = theta.copy()
    dy = theta.copy()
    dx[0] += scale
    dy[1] += scale
    if vehicle_num == 1:
        if sens:
            V_dx, _ = uav_sens(m, theta, dx)
            V_dy, _ = uav_sens(m, theta, dy)
        else:
            _, V_dx = uav_solve(t, obs, dx, ref, nfe_num=nfe_num)
            _, V_dy = uav_solve(t, obs, dy, ref, nfe_num=nfe_num)
    else:
        if sens:
            V_dx, _ = ugv_sens(m, theta, dx)
            V_dy, _ = ugv_sens(m, theta, dy)
        else:
            _, V_dx = ugv_solve(t, obs, dx, ref, nfe_num=nfe_num)
            _, V_dy = ugv_solve(t, obs, dy, ref, nfe_num=nfe_num)
    dJ_dx = (V_dx - V) / scale
    dJ_dy = (V_dy - V) / scale
    g = np.array([dJ_dx, dJ_dy])

    if iter_num != 0 and np.linalg.norm(TR.p) != 0:
        s_k = np.reshape(p, (-1, 1))
        y_k = np.reshape(g - g_prev, (-1, 1))
        s_k_T = np.transpose(s_k)
        y_k_T = np.transpose(y_k)
        B_T = np.transpose(B)

        if y_k_T @ s_k > 0:
            B = B + (y_k @ y_k_T) / (y_k_T @ s_k) - (B @ s_k @ s_k_T @ B_T) / (s_k_T @ B @ s_k)

    return m, V, g, B, traj


def vehicle_warm_start(m_1, theta_1, TR_1, m_2, theta_2, TR_2):
    _, V_1_new, _ = uav_warm_start(m_1, np.concatenate((theta_1[0:2]+TR_1.p, theta_1[2:4])))
    _, V_2_new, _ = ugv_warm_start(m_2, np.concatenate((theta_2[0:2]+TR_2.p, theta_2[2:4])))

    return V_1_new, V_2_new


def uav_solve(t_g, obs_g, theta, ref, nfe_num=10, solve_display_mode=False):
    if ref is None:
        ref = linear_init_uav(t_g, obs_g, theta)

    model_g, _ = decentralized_uav(t_g, obs_g, ref, theta, nfe_num)

    m_g, flag_g = solve(model_g, tee=solve_display_mode)

    # save result
    ref_traj = SaveUAV()
    if flag_g == 'optimal':
        ref_traj.store_opt(m_g)
        V = value(m_g.V)
    else:
        V, ref_traj = 1e8, ref_traj.reset()

    return m_g, V, ref_traj


def uav_warm_start(m, new_theta, solve_display_mode=False):
    m.x_f = new_theta[0]
    m.y_f = new_theta[1]

    m.ipopt_zL_in.update(m.ipopt_zL_out)
    m.ipopt_zU_in.update(m.ipopt_zU_out)

    # solve OCP problem
    m_g, flag_g = solve(m, tee=solve_display_mode, ws=True)

    # save result
    ref_traj = SaveUAV()
    if flag_g == 'optimal':
        ref_traj.store_opt(m_g)
        V = value(m_g.V)
    else:
        V, ref_traj = 1e8, ref_traj.reset()

    return m_g, V, ref_traj


def uav_sens(m, theta, pert_theta):
    m.x_f = theta[0]
    m.y_f = theta[1]
    paramList = [m.x_f, m.y_f]
    pertList = [pert_theta[0], pert_theta[1]]

    sens = SensitivityInterface(m, clone_model=True)
    sens.setup_sensitivity(paramList)

    m_sipopt = sens.model_instance
    sens.perturb_parameters(pertList)

    ipopt_sens = SolverFactory('ipopt_sens', solver_io='nl')
    ipopt_sens.options['run_sens'] = 'yes'
    m_sipopt, flag_g, _ = sipopt_solve(m_sipopt, ipopt_sens)

    # save result
    ref_traj = SaveUAV()
    if flag_g == 'optimal':
        ref_traj.store_sens_opt(m_sipopt)
        x = m_sipopt.sens_sol_state_1[m_sipopt.x[1]]
        y = m_sipopt.sens_sol_state_1[m_sipopt.y[1]]
        sum_u = m_sipopt.sens_sol_state_1[m_sipopt.sum_u[1]]
        V = (x - pert_theta[0]) ** 2 + \
            (y - pert_theta[1]) ** 2 + \
            sum_u
    else:
        V, ref_traj = 1e8, ref_traj.reset()

    return V, ref_traj


def ugv_solve(t_g, obs_g, theta, ref, nfe_num=10, solve_display_mode=False):
    if ref is None:
        ref = linear_init_ugv(t_g, obs_g, theta)

    model_g, _ = decentralized_ugv(t_g, obs_g, ref, theta, nfe_num)

    m_g, flag_g = solve(model_g, tee=solve_display_mode)

    ref_traj = SaveUGV()
    if flag_g == 'optimal':
        ref_traj.store_opt(m_g)
        V = value(m_g.V)
    else:
        V, ref_traj = 1e8, ref_traj.reset()

    return m_g, V, ref_traj


def ugv_warm_start(m, new_theta, solve_display_mode=False):
    m.x_f = new_theta[0]
    m.y_f = new_theta[1]

    m.ipopt_zL_in.update(m.ipopt_zL_out)
    m.ipopt_zU_in.update(m.ipopt_zU_out)

    m_g, flag_g = solve(m, tee=solve_display_mode, ws=True)

    ref_traj = SaveUGV()
    if flag_g == 'optimal':
        ref_traj.store_opt(m_g)
        V = value(m_g.V)
    else:
        V, ref_traj = 1e8, ref_traj.reset()

    return m_g, V, ref_traj


def ugv_sens(m, theta, pert_theta):
    m.x_f = theta[0]
    m.y_f = theta[1]
    paramList = [m.x_f, m.y_f]
    pertList = [pert_theta[0], pert_theta[1]]

    sens = SensitivityInterface(m, clone_model=True)
    sens.setup_sensitivity(paramList)

    m_sipopt = sens.model_instance
    sens.perturb_parameters(pertList)

    ipopt_sens = SolverFactory('ipopt_sens', solver_io='nl')
    ipopt_sens.options['run_sens'] = 'yes'
    m_sipopt, flag_g, _ = sipopt_solve(m_sipopt, ipopt_sens)

    # save result
    ref_traj = SaveUGV()
    if flag_g == 'optimal':
        ref_traj.store_sens_opt(m_sipopt)
        x = m_sipopt.sens_sol_state_1[m_sipopt.x[1]]
        y = m_sipopt.sens_sol_state_1[m_sipopt.y[1]]
        sum_u = m_sipopt.sens_sol_state_1[m_sipopt.sum_u[1]]
        V = (x - pert_theta[0]) ** 2 + \
            (y - pert_theta[1]) ** 2 + \
            sum_u
    else:
        V, ref_traj = 1e8, ref_traj.reset()

    return V, ref_traj
