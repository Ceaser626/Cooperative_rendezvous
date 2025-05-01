import numpy as np
from scipy import signal
from Env.UAV_three_dof_ode import equations_of_motion
from Env.save_tools import SaveUAV


class UAV:

    def __init__(self):
        # Initial state
        self.x_i = -20
        self.y_i = -5
        self.h_i = 100
        self.v_i = 15
        self.eps_i = np.radians(10)
        self.gamma_i = -np.radians(10)
        # env parameter
        self.t = 0
        self.dt = 1
        self.h = 0.05
        self.count = 0
        self.save_status = SaveUAV()
        # mission parameter
        self.T_min = 0
        self.T_max = 2
        self.phi_min = -np.radians(24)
        self.phi_max = np.radians(24)
        self.CL_min = -0.7
        self.CL_max = 0.7
        # Wind
        t_p_len = int(100 / self.h)
        self.t_p = np.zeros(t_p_len)
        self.u_disturb = np.zeros(t_p_len)
        self.v_disturb = np.zeros(t_p_len)
        self.w_disturb = np.zeros(t_p_len)

    def reset(self, o=None, dryden=False, save_dryden=False):

        if dryden:
            self.t_p, self.u_disturb, self.v_disturb, self.w_disturb = self.dryden_wind(save=save_dryden)

        if o is not None:
            x = o[0]
            y = o[1]
            h = o[2]
            v = o[3]
            eps = o[4]
            gamma = o[5]
        else:
            x = self.x_i
            y = self.y_i
            h = self.h_i
            v = self.v_i
            eps = self.eps_i
            gamma = self.gamma_i

        self.t = 0
        self.count = 0
        self.save_status.reset()

        self.observation = [x, y, h, v, eps, gamma]
        o = self.observation

        return np.array(o, dtype=np.float32)

    def step(self, a):
        clipped_a = self.clip_action(a)

        if self.t == 0:
            self.save_status.store_sim(self.t, self.observation, clipped_a)

        int_num = int(self.dt / self.h)
        done = False
        for i in range(int_num):
            action = clipped_a

            j = self.count
            disturb_wind = [self.u_disturb[j], self.v_disturb[j], self.w_disturb[j]]

            k1 = equations_of_motion(self.observation, action, disturb_wind)
            k2 = equations_of_motion(self.observation + self.h * k1 / 2, action, disturb_wind)
            k3 = equations_of_motion(self.observation + self.h * k2 / 2, action, disturb_wind)
            k4 = equations_of_motion(self.observation + self.h * k3, action, disturb_wind)
            self.observation = self.observation + self.h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

            self.t = self.t + self.h
            self.count = self.count + 1
            self.save_status.store_sim(self.t, self.observation, action)  # save current t, o, a

            if self.t >= 1000 or self.observation[2] <= 0.01:
                done = True
                break

        o = self.observation

        if np.isnan(self.observation).any() or np.any(self.observation > 1e6):
            o = [0] * 6
            done = True
            return np.array(o, dtype=np.float32), done, self.save_status

        return np.array(o, dtype=np.float32), done, self.save_status


    def clip_action(self, a):
        low_bound = [self.T_min] + [self.phi_min] + [self.CL_min]
        high_bound = [self.T_max] + [self.phi_max] + [self.CL_max]
        clipped_a = np.clip(a, low_bound, high_bound)

        return clipped_a

    def dryden_wind(self, w_20=30, save=False):
        w_20 = w_20 * 0.514444
        altitude = 200
        air_speed = 20

        L_u = altitude / ((0.177 + 0.000823 * altitude) ** 1.2)
        L_v = L_u
        L_w = altitude
        sigma_w = 0.1 * w_20
        sigma_u = sigma_w / ((0.177 + 0.000823 * altitude) ** 0.4)
        sigma_v = sigma_u

        num_u = [sigma_u * np.sqrt(2 * L_u / np.pi / air_speed) * air_speed]
        den_u = [L_u, air_speed]
        H_u = signal.TransferFunction(num_u, den_u)

        b = sigma_v * np.sqrt(L_v / np.pi / air_speed)
        num_v = [np.sqrt(3) * L_v / air_speed * b, b]
        den_v = [(L_v / air_speed) ** 2, 2 * L_v / air_speed, 1]
        H_v = signal.TransferFunction(num_v, den_v)

        c = sigma_w * np.sqrt(L_w / np.pi / air_speed)
        num_w = [np.sqrt(3) * L_w / air_speed * c, c]
        den_w = [(L_w / air_speed) ** 2, 2 * L_w / air_speed, 1]
        H_w = signal.TransferFunction(num_w, den_w)

        t_end = int(100)
        num_samples = int(t_end / self.h)
        t_p = np.linspace(0, t_end, num_samples)
        mean = 0
        std = 1
        wgn_input_u = np.random.normal(mean, std, size=num_samples)
        wgn_input_v = np.random.normal(mean, std, size=num_samples)
        wgn_input_w = np.random.normal(mean, std, size=num_samples)

        tout1, u_disturb, x1 = signal.lsim(H_u, wgn_input_u, t_p)
        tout2, v_disturb, x2 = signal.lsim(H_v, wgn_input_v, t_p)
        tout3, w_disturb, x3 = signal.lsim(H_w, wgn_input_w, t_p)

        # save
        if save:
            t_save = np.array(t_p, dtype=object)
            u_save = np.array(u_disturb, dtype=object)
            v_save = np.array(v_disturb, dtype=object)
            w_save = np.array(w_disturb, dtype=object)
            np.savez(f'Save/dryden_result', t_save=t_save, u_save=u_save, v_save=v_save, w_save=w_save)

        return t_p, u_disturb, v_disturb, w_disturb
