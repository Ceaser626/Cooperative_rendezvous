import numpy as np
from Env.UGV_three_dof_ode import equations_of_motion
from Env.save_tools import SaveUGV


class UGV:

    def __init__(self):
        # Initial state
        self.x_i = 0
        self.y_i = 0
        self.v_i = 10
        self.a_i = 0
        self.eps_i = 0
        # env parameter
        self.t = 0
        self.dt = 1
        self.h = 0.05
        self.count = 0
        self.save_status = SaveUGV()
        # mission parameter
        self.a_c_min = -5
        self.a_c_max = 5
        self.eps_c_min = -np.radians(30)
        self.eps_c_max = np.radians(30)

    def reset(self, o=None):
        if o is not None:
            x = o[0]
            y = o[1]
            v = o[2]
            a = o[3]
            eps = o[4]
        else:
            x = self.x_i
            y = self.y_i
            v = self.v_i
            a = self.a_i
            eps = self.eps_i

        self.t = 0
        self.count = 0
        self.save_status.reset()

        self.observation = [x, y, v, a, eps]
        o = self.observation

        return np.array(o, dtype=np.float32)

    def step(self, a, t_f=200):
        clipped_a = self.clip_action(a)

        if self.t == 0:
            self.save_status.store_sim(self.t, self.observation, clipped_a)

        int_num = int(self.dt / self.h)
        done = False
        for i in range(int_num):
            action = clipped_a

            k1 = equations_of_motion(self.observation, action)
            k2 = equations_of_motion(self.observation + self.h * k1 / 2, action)
            k3 = equations_of_motion(self.observation + self.h * k2 / 2, action)
            k4 = equations_of_motion(self.observation + self.h * k3, action)
            self.observation = self.observation + self.h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

            self.t = self.t + self.h
            self.count = self.count + 1
            self.save_status.store_sim(self.t, self.observation, action)  # save current t, o, a

            if self.t >= t_f:
                done = True
                break

        o = self.observation

        if np.isnan(self.observation).any() or np.any(self.observation > 1e6):
            o = [0] * 5
            done = True
            return np.array(o, dtype=np.float32), done, self.save_status

        return np.array(o, dtype=np.float32), done, self.save_status


    def clip_action(self, a):
        low_bound = [self.a_c_min] + [self.eps_c_min]
        high_bound = [self.a_c_max] + [self.eps_c_max]
        clipped_a = np.clip(a, low_bound, high_bound)

        return clipped_a
