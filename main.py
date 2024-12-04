import numpy as np
import random

from Algorithm.rendezvous_guidance import theta_traj_update
from Problem.estimate_module import estimate_obs
from Problem.control_module import uav_control, ugv_control
from Env.env_uav import UAV
from Env.env_ugv import UGV
from Utils.tools import interpolate


def sim(num=1, nfe_point=10, save=True, sens=True, closed_loop=False, disturb=False, thresh=1e-1):
    random.seed(num)
    np.random.seed(num)
    theta_list = np.array([[], [], [], []])
    V_list = np.array([])

    # env
    sim_uav = UAV()
    sim_ugv = UGV()
    # parameters
    nfe_num = nfe_point
    debug_mode = False
    sens_and_warm_start = sens
    TR_para_dict = {'delta_init': 20, 'delta_max': 100, 'eta': 0.05, 'threshold': thresh, 'max_iter': 50}

    frequency_guide = 0.1  # 0.1Hz
    frequency_control = 1  # 1Hz
    guidance_horizon = int(1 / frequency_guide)
    control_horizon = int(1 / frequency_control)
    control_loop_num = int(frequency_control / frequency_guide)
    sim_uav.dt = 1 / frequency_control
    sim_uav.frequency_guide = frequency_guide
    sim_uav.frequency_control = frequency_control
    sim_ugv.dt = 1 / frequency_control
    sim_ugv.frequency_guide = frequency_guide
    sim_ugv.frequency_control = frequency_control
    _ = sim_uav.reset(dryden=disturb, save_dryden=save)
    _ = sim_ugv.reset()

    theta_1 = np.array([150, 80, np.radians(0), 35])
    theta_2 = np.array([200, 20, np.radians(0), 35])
    t_est_g, obs_est_p, obs_est_s = sim_uav.t, sim_uav.observation, sim_ugv.observation
    theta_1, ref_traj_1, theta_2, ref_traj_2, iter_history = theta_traj_update(t_est_g, obs_est_p, obs_est_s, theta_1, theta_2, TR_para_dict, nfe_num=nfe_num, sens=sens_and_warm_start)
    theta_list = np.append(theta_list, np.array([[theta_1[0]], [theta_1[1]], [theta_2[0]], [theta_2[1]]]), axis=1)
    V_list = np.append(V_list, np.array([0.5*ref_traj_1.sum_u + 0.5*ref_traj_2.sum_u]))
    obs_last_1_g, obs_last_2_g = sim_uav.observation, sim_ugv.observation

    if save:
        iter_history.save(f'TR_iteration_{sens}')
        ref_traj_1.save('dec_traj_1')
        ref_traj_2.save('dec_traj_2')

    if closed_loop:
        print(f'Control_loop: 0')
        t_est_c, obs_est_1_c, obs_est_2_c = sim_uav.t, sim_uav.observation, sim_ugv.observation
        theta_1_now, theta_2_now = theta_1, theta_2
        ref_traj_1_now, ref_traj_2_now = ref_traj_1, ref_traj_2
        action_1, traj_1_c = uav_control(t_est_c, obs_est_1_c, theta_1_now, ref_traj_1_now, nfe_num=nfe_num)
        action_2, traj_2_c = ugv_control(t_est_c, obs_est_2_c, theta_2_now, ref_traj_2_now, nfe_num=nfe_num)
        obs_last_1_c, obs_last_2_c = sim_uav.observation, sim_ugv.observation

        # -------------------------------------------------------------------------------------------------- #
        # ---------------------------------------- Simulation Start ---------------------------------------- #
        g_I = 0
        c_I = 0
        while True:
            for i in range(control_loop_num):
                next_obs_1, done, sim_traj_1 = sim_uav.step(action_1)
                if done:
                    next_obs_2, _, sim_traj_2 = sim_ugv.step(action_2, t_f=sim_uav.t)
                else:
                    next_obs_2, _, sim_traj_2 = sim_ugv.step(action_2)
                obs_1, obs_2 = next_obs_1, next_obs_2
                r_x_1, r_y_1, h_1 = obs_1[0], obs_1[1], obs_1[2]
                r_x_2, r_y_2 = obs_2[0], obs_2[1]
                r_err, h_err = np.sqrt((r_x_1 - r_x_2) ** 2 + (r_y_1 - r_y_2) ** 2), abs(h_1)
                print(f'Position error: {r_err}, Altitude error: {h_err}')
                if done:
                    break

                t_est_c = sim_uav.t
                obs_est_1_c, obs_est_2_c = estimate_obs(obs_last_1_c, obs_last_2_c, traj_1_c, traj_2_c, t_est_c, control_horizon)

                c_I += 1
                print(f'Control_loop: {c_I}')
                if sim_uav.t <= theta_1[-1]:
                    action_1, traj_1_c = uav_control(t_est_c, obs_est_1_c, theta_1_now, ref_traj_1_now, nfe_num=nfe_num)
                    action_2, traj_2_c = ugv_control(t_est_c, obs_est_2_c, theta_2_now, ref_traj_2_now, nfe_num=nfe_num)
                    obs_last_1_c, obs_last_2_c = sim_uav.observation, sim_ugv.observation

            if save:
                sim_traj_1.save(f'sim_dec_traj_1')
                sim_traj_2.save(f'sim_dec_traj_2')
                theta_list = np.array(theta_list, dtype=object)
                V_list = np.array(V_list, dtype=object)
                np.savez(f'Save/opt_profile', theta_list=theta_list, J_list=V_list)
            if done:
                print(f'End')
                print(V_list)
                break

            t_est_g = sim_uav.t
            obs_est_p, obs_est_s = estimate_obs(obs_last_1_g, obs_last_2_g, ref_traj_1, ref_traj_2, t_est_g, guidance_horizon)

            g_I += 1
            print(f'Guidance_loop: {g_I}')
            try:
                theta_1, ref_traj_1, theta_2, ref_traj_2, iter_history = theta_traj_update(t_est_g, obs_est_p, obs_est_s, theta_1_now, theta_2_now, TR_para_dict,
                                                                                           traj_p=ref_traj_1_now, traj_s=ref_traj_2_now, nfe_num=nfe_num, sens=sens_and_warm_start)
                theta_list = np.append(theta_list, np.array([[theta_1[0]], [theta_1[1]], [theta_2[0]], [theta_2[1]]]), axis=1)
                V_list = np.append(V_list, np.array([0.5*ref_traj_1.sum_u + 0.5*ref_traj_2.sum_u]))
            except:
                continue
            obs_last_1_g, obs_last_2_g = sim_uav.observation, sim_ugv.observation


if __name__ == "__main__":
    sim()
    sim(closed_loop=True, disturb=True)
