import numpy as np
import sympy as sp
from pyomo.environ import *
from scipy.interpolate import interp1d


def lgr_points(num):
    # Flipped shifted-Radau points
    if num == 2:
        tau_list = [0.333333333333333, 1]
    elif num == 3:
        tau_list = [0.155051025721682, 0.644948974278318, 1]
    elif num == 4:
        tau_list = [0.088587959512704, 0.409466864440735, 0.787659461760847, 1]
    elif num == 5:
        tau_list = [0.057104196114518, 0.276843013638124, 0.583590432368917, 0.860240135656220, 1]
    elif num == 6:
        tau_list = [0.039809857051469, 0.198013417873608, 0.437974810247386, 0.695464273353636, 0.901464914201174, 1]
    else:
        tau_list = []

    return tau_list


class SaveError:

    def __init__(self):
        self.t = []
        self.r_err = []
        self.psi_err = []

    def store(self, t, r, psi):
        self.t.append(t)
        self.r_err.append(r)
        self.psi_err.append(psi)

    def save(self, save_name):
        t = np.array(self.t, dtype=object)
        r_err = np.array(self.r_err, dtype=object)
        psi_err = np.array(self.psi_err, dtype=object)
        np.savez(f'save/{save_name}', t=t, r_err=r_err, psi_err=psi_err)


class SaveBaseline:

    def __init__(self):
        self.t = []
        self.x = []
        self.y = []
        self.z = []
        self.psi = []
        self.psi_dot = []
        self.delta_a = []
        self.delta_l = []
        self.delta_r = []

    def reset(self):
        self.t = []
        self.x = []
        self.y = []
        self.z = []
        self.psi = []
        self.psi_dot = []
        self.delta_a = []
        self.delta_l = []
        self.delta_r = []

    def store_baseline_guidance(self, m):
        for i in m.tau:
            self.t.append(value(m.time[i]))
            self.x.append(value(m.x[i]))
            self.y.append(value(m.y[i]))
            self.z.append(value(m.z[i]))
            self.psi.append(value(m.psi[i]))
            self.psi_dot.append(value(m.psi_dot[i]))

    def store_baseline_control(self, m):
        for i in m.tau:
            self.t.append(value(m.time[i]))
            self.psi.append(value(m.psi[i]))
            self.psi_dot.append(value(m.psi_dot[i]))
            self.delta_a.append(value(m.delta_a[i]))
            if value(m.delta_a[i]) > 0:
                delta_l = 0
                delta_r = value(m.delta_a[i])
            else:
                delta_l = - value(m.delta_a[i])
                delta_r = 0
            self.delta_l.append(delta_l)
            self.delta_r.append(delta_r)
        # interpolate at t=0
        self.delta_a[0] = nlp_u_interpolate(self.t[0], [self.t[1], self.t[2]], [self.delta_a[1], self.delta_a[2]], 2)

    def store_iter(self, scene):  # scene is a complete trajectory
        self.t.append(scene.t)
        self.x.append(scene.x)
        self.y.append(scene.y)
        self.z.append(scene.z)
        self.psi.append(scene.psi)
        self.psi_dot.append(scene.psi_dot)

    def save(self, save_name):
        t = np.array(self.t, dtype=object)
        x = np.array(self.x, dtype=object)
        y = np.array(self.y, dtype=object)
        z = np.array(self.z, dtype=object)
        psi = np.array(self.psi, dtype=object)
        psi_dot = np.array(self.psi_dot, dtype=object)
        delta_a = np.array(self.delta_a, dtype=object)
        np.savez(f'save/{save_name}', t=t, r_x=x, r_y=y, r_z=z,
                 psi=psi, psi_dot=psi_dot, delta_a=delta_a)


class SaveWind:

    def __init__(self):
        self.t_p = []
        self.w_x = []
        self.w_y = []
        self.w_z = []
        self.u_disturb = []
        self.v_disturb = []
        self.w_disturb = []

    def store_env(self, env):
        self.t_p = env.t_p
        self.w_x = env.w_x
        self.w_y = env.w_y
        self.w_z = env.w_z
        self.u_disturb = env.u_disturb
        self.v_disturb = env.v_disturb
        self.w_disturb = env.w_disturb

    def store_mc(self, scene):
        self.t_p.append(scene.t_p)
        self.w_x.append(scene.w_x)
        self.w_y.append(scene.w_y)
        self.w_z.append(scene.w_z)
        self.u_disturb.append(scene.u_disturb)
        self.v_disturb.append(scene.v_disturb)
        self.w_disturb.append(scene.w_disturb)

    def save(self, save_name):
        t_p = np.array(self.t_p, dtype=object)
        w_x = np.array(self.w_x, dtype=object)
        w_y = np.array(self.w_y, dtype=object)
        w_z = np.array(self.w_z, dtype=object)
        u_disturb = np.array(self.u_disturb, dtype=object)
        v_disturb = np.array(self.v_disturb, dtype=object)
        w_disturb = np.array(self.w_disturb, dtype=object)
        np.savez(f'save/{save_name}', t_p=t_p, w_x=w_x, w_y=w_y, w_z=w_z,
                 u_disturb=u_disturb, v_disturb=v_disturb, w_disturb=w_disturb)


def interpolate(t_list, value_list):
    f_value = interp1d(t_list, value_list, kind='linear', fill_value="extrapolate")
    return f_value


def nlp_u_interpolate(t, t_list, value_list, num):
    u_t = 0
    for j in range(num):
        cof = 1
        for k in range(num):
            if k != j:
                cof = cof * (t - t_list[k]) / (t_list[j] - t_list[k])
        u_t = u_t + cof * value_list[j]

    return u_t


def sp_norm(v):
    output = sp.sqrt(v[0, 0]**2 + v[1, 0]**2 + v[2, 0]**2)
    return output


def np_norm(v):
    output = np.sqrt(v[0, 0]**2 + v[1, 0]**2 + v[2, 0]**2)
    return output
