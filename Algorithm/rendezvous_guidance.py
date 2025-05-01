import numpy as np
from Algorithm.trust_region import TrustRegionOpt
from Env.save_tools import SaveIteration
from Problem.guidance_module import vehicle_opt, vehicle_warm_start


def theta_traj_update(t, obs_p, obs_s, theta_p, theta_s, TR_para_dict,
                      traj_p=None, traj_s=None, wind=None, estimator=None, nfe_num=10, sens=False):
    # Instantiation
    TR_p = TrustRegionOpt(TR_para_dict)
    TR_s = TrustRegionOpt(TR_para_dict)
    iter_history = SaveIteration()
    # Termination judge
    terminate = False
    threshold = TR_para_dict['threshold']
    max_iter = TR_para_dict['max_iter']

    while not terminate:
        if TR_p.iter_num == 0 or np.linalg.norm(TR_p.p) != 0:
            m_p, V_p, g_p, B_p, traj_p = vehicle_opt(t, obs_p, theta_p, TR_p, traj_p, vehicle_num=1, nfe_num=nfe_num, sens=sens, wind=wind, estimator=estimator)
        if TR_s.iter_num == 0 or np.linalg.norm(TR_s.p) != 0:
            m_s, V_s, g_s, B_s, traj_s = vehicle_opt(t, obs_s, theta_s, TR_s, traj_s, vehicle_num=2, nfe_num=nfe_num, sens=sens, wind=wind, estimator=estimator)

        TR_p.update_info(theta_p[0:2], V_p, g_p, B_p, theta_s[0:2], V_s, g_s, B_s)
        TR_s.update_info(theta_s[0:2], V_s, g_s, B_s, theta_p[0:2], V_p, g_p, B_p)

        TR_p.determine_p()
        TR_s.determine_p()

        V_p_new, V_s_new = vehicle_warm_start(m_p, theta_p, TR_p, m_s, theta_s, TR_s)
        theta_p_new = theta_p.copy()
        theta_s_new = theta_s.copy()
        theta_p_new[0:2] = TR_p.update_region_size(theta_p[0:2], V_p_new, theta_s[0:2], V_s, g_s, B_s)
        theta_s_new[0:2] = TR_s.update_region_size(theta_s[0:2], V_s_new, theta_p[0:2], V_p, g_p, B_p)
        current_theta_d = np.linalg.norm(theta_p - theta_s)
        theta_p = theta_p_new.copy()
        theta_s = theta_s_new.copy()

        iter_history.store_info(traj_p, TR_p, 1)
        iter_history.store_info(traj_s, TR_s, 2)

        print(f'Iteration: {TR_p.iter_num}\n'
              f'theta_p: {theta_p[0:2]}, V_p: {TR_p.V}, dV_sum_p: {np.linalg.norm(TR_p.g)}\n'
              f'theta_s: {theta_s[0:2]}, V_s: {TR_s.V}, dV_sum_s: {np.linalg.norm(TR_s.g)}')
        TR_p.V_prev = TR_p.V
        TR_s.V_prev = TR_s.V
        TR_p.g_prev = g_p
        TR_s.g_prev = g_s

        if 0 < np.linalg.norm(TR_p.g) <= threshold and 0 < np.linalg.norm(TR_s.g) <= threshold and current_theta_d <= 1e-1:
            terminate = True
        if TR_p.iter_num >= max_iter:
            terminate = True

        # iter_history.save('TR_iteration')

    return theta_p, traj_p, theta_s, traj_s, iter_history
