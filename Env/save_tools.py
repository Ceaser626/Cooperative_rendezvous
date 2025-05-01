import numpy as np
from pyomo.environ import *
from Utils.tools import nlp_u_interpolate


class SaveUAV:

    def __init__(self):
        self.reset()

    def reset(self):
        self.t = []
        self.x = []
        self.y = []
        self.h = []
        self.v = []
        self.eps = []
        self.gamma = []
        self.T = []
        self.phi = []
        self.CL = []
        self.sum_u = 0

    def store_sim(self, t, o, control):
        x, y, h, v, eps, gamma = o
        T, phi, CL = control

        self.t.append(t)
        self.x.append(x)
        self.y.append(y)
        self.h.append(h)
        self.v.append(v)
        self.eps.append(eps)
        self.gamma.append(gamma)
        self.T.append(T)
        self.phi.append(phi)
        self.CL.append(CL)

    def store_opt(self, m, number=None, store_control=False):
        if number is None:
            for i in m.tau:
                self.t.append(value(m.time[i]))
                self.x.append(value(m.x[i]))
                self.y.append(value(m.y[i]))
                self.h.append(value(m.h[i]))
                self.v.append(value(m.v[i]))
                self.eps.append(value(m.eps[i]))
                self.gamma.append(value(m.gamma[i]))
                self.T.append(value(m.T[i]))
                self.phi.append(value(m.phi[i]))
                self.CL.append(value(m.CL[i]))
        elif number == 1:
            for i in m.tau:
                self.t.append(value(m.time[i]))
                self.x.append(value(m.x_1[i]))
                self.y.append(value(m.y_1[i]))
                self.h.append(value(m.h_1[i]))
                self.v.append(value(m.v_1[i]))
                self.eps.append(value(m.eps_1[i]))
                self.gamma.append(value(m.gamma_1[i]))
                self.T.append(value(m.T_1[i]))
                self.phi.append(value(m.phi_1[i]))
                self.CL.append(value(m.CL_1[i]))
        else:
            for i in m.tau:
                self.t.append(value(m.time[i]))
                self.x.append(value(m.x_2[i]))
                self.y.append(value(m.y_2[i]))
                self.h.append(value(m.h_2[i]))
                self.v.append(value(m.v_2[i]))
                self.eps.append(value(m.eps_2[i]))
                self.gamma.append(value(m.gamma_2[i]))
                self.T.append(value(m.T_2[i]))
                self.phi.append(value(m.phi_2[i]))
                self.CL.append(value(m.CL_2[i]))
        if not store_control:
            self.sum_u = value(m.sum_u[1])

        if len(self.t) >= 3:
            self.T[0] = nlp_u_interpolate(self.t[0], [self.t[1], self.t[2]], [self.T[1], self.T[2]], 2)
            self.phi[0] = nlp_u_interpolate(self.t[0], [self.t[1], self.t[2]], [self.phi[1], self.phi[2]], 2)
            self.CL[0] = nlp_u_interpolate(self.t[0], [self.t[1], self.t[2]], [self.CL[1], self.CL[2]], 2)

    def store_sens_opt(self, m, opt_time=None):
        for i in m.tau:
            if i == 0:
                self.t.append(value(m.time[i]))
                self.x.append(value(m.x[i]))
                self.y.append(value(m.y[i]))
                self.h.append(value(m.h[i]))
                self.v.append(value(m.v[i]))
                self.eps.append(value(m.eps[i]))
                self.gamma.append(value(m.gamma[i]))
                self.T.append(value(m.T[i]))
                self.phi.append(value(m.phi[i]))
                self.CL.append(value(m.CL[i]))
            else:
                self.t.append(m.sens_sol_state_1[m.time[i]])
                self.x.append(m.sens_sol_state_1[m.x[i]])
                self.y.append(m.sens_sol_state_1[m.y[i]])
                self.h.append(m.sens_sol_state_1[m.h[i]])
                self.v.append(m.sens_sol_state_1[m.v[i]])
                self.eps.append(m.sens_sol_state_1[m.eps[i]])
                self.gamma.append(m.sens_sol_state_1[m.gamma[i]])
                self.T.append(m.sens_sol_state_1[m.T[i]])
                self.phi.append(m.sens_sol_state_1[m.phi[i]])
                self.CL.append(m.sens_sol_state_1[m.CL[i]])

        if len(self.t) >= 3:
            self.T[0] = nlp_u_interpolate(self.t[0], [self.t[1], self.t[2]], [self.T[1], self.T[2]], 2)
            self.phi[0] = nlp_u_interpolate(self.t[0], [self.t[1], self.t[2]], [self.phi[1], self.phi[2]], 2)
            self.CL[0] = nlp_u_interpolate(self.t[0], [self.t[1], self.t[2]], [self.CL[1], self.CL[2]], 2)

    def store_mc(self, scene):
        self.t.append(scene.t)
        self.x.append(scene.x)
        self.y.append(scene.y)
        self.h.append(scene.h)
        self.v.append(scene.v)
        self.eps.append(scene.eps)
        self.gamma.append(scene.gamma)
        self.T.append(scene.T)
        self.phi.append(scene.phi)
        self.CL.append(scene.CL)

    def save(self, save_name):
        t = np.array(self.t, dtype=object)
        x = np.array(self.x, dtype=object)
        y = np.array(self.y, dtype=object)
        h = np.array(self.h, dtype=object)
        v = np.array(self.v, dtype=object)
        eps = np.array(self.eps, dtype=object)
        gamma = np.array(self.gamma, dtype=object)
        T = np.array(self.T, dtype=object)
        phi = np.array(self.phi, dtype=object)
        CL = np.array(self.CL, dtype=object)
        np.savez(f'Save/{save_name}', t=t, x=x, y=y, h=h, v=v, eps=eps, gamma=gamma,
                 T=T, phi=phi, CL=CL)

    def load(self, save_name):
        data = np.load(f'Save/{save_name}.npz', allow_pickle=True)
        self.t = data['t']
        self.x = data['x']
        self.y = data['y']
        self.h = data['h']
        self.v = data['v']
        self.eps = data['eps']
        self.gamma = data['gamma']
        self.T = data['T']
        self.phi = data['phi']
        self.CL = data['CL']


class SaveUGV:

    def __init__(self):
        self.reset()

    def reset(self):
        self.t = []
        self.x = []
        self.y = []
        self.v = []
        self.a = []
        self.eps = []
        self.a_c = []
        self.eps_c = []
        self.sum_u = 0

    def store_sim(self, t, o, control):
        x, y, v, a, eps = o
        a_c, eps_c = control

        self.t.append(t)
        self.x.append(x)
        self.y.append(y)
        self.v.append(v)
        self.a.append(a)
        self.eps.append(eps)
        self.a_c.append(a_c)
        self.eps_c.append(eps_c)

    def store_opt(self, m, number=None, store_control=False):
        if number is None:
            for i in m.tau:
                self.t.append(value(m.time[i]))
                self.x.append(value(m.x[i]))
                self.y.append(value(m.y[i]))
                self.v.append(value(m.v[i]))
                self.a.append(value(m.a[i]))
                self.eps.append(value(m.eps[i]))
                self.a_c.append(value(m.a_c[i]))
                self.eps_c.append(value(m.eps_c[i]))
        elif number == 1:
            for i in m.tau:
                self.t.append(value(m.time[i]))
                self.x.append(value(m.x_1[i]))
                self.y.append(value(m.y_1[i]))
                self.v.append(value(m.v_1[i]))
                self.a.append(value(m.a_1[i]))
                self.eps.append(value(m.eps_1[i]))
                self.a_c.append(value(m.a_c_1[i]))
                self.eps_c.append(value(m.eps_c_1[i]))
        else:
            for i in m.tau:
                self.t.append(value(m.time[i]))
                self.x.append(value(m.x_2[i]))
                self.y.append(value(m.y_2[i]))
                self.v.append(value(m.v_2[i]))
                self.a.append(value(m.a_2[i]))
                self.eps.append(value(m.eps_2[i]))
                self.a_c.append(value(m.a_c_2[i]))
                self.eps_c.append(value(m.eps_c_2[i]))
        if not store_control:
            self.sum_u = value(m.sum_u[1])

        if len(self.t) >= 3:
            self.a_c[0] = nlp_u_interpolate(self.t[0], [self.t[1], self.t[2]], [self.a_c[1], self.a_c[2]], 2)
            self.eps_c[0] = nlp_u_interpolate(self.t[0], [self.t[1], self.t[2]], [self.eps_c[1], self.eps_c[2]], 2)

    def store_sens_opt(self, m, opt_time=None):
        for i in m.tau:
            if i == 0:
                self.t.append(value(m.time[i]))
                self.x.append(value(m.x[i]))
                self.y.append(value(m.y[i]))
                self.v.append(value(m.v[i]))
                self.a.append(value(m.a[i]))
                self.eps.append(value(m.eps[i]))
                self.a_c.append(value(m.a_c[i]))
                self.eps_c.append(value(m.eps_c[i]))
            else:
                self.t.append(m.sens_sol_state_1[m.time[i]])
                self.x.append(m.sens_sol_state_1[m.x[i]])
                self.y.append(m.sens_sol_state_1[m.y[i]])
                self.v.append(m.sens_sol_state_1[m.v[i]])
                self.a.append(m.sens_sol_state_1[m.a[i]])
                self.eps.append(m.sens_sol_state_1[m.eps[i]])
                self.a_c.append(m.sens_sol_state_1[m.a_c[i]])
                self.eps_c.append(m.sens_sol_state_1[m.eps_c[i]])

        if len(self.t) >= 3:
            self.a_c[0] = nlp_u_interpolate(self.t[0], [self.t[1], self.t[2]], [self.a_c[1], self.a_c[2]], 2)
            self.eps_c[0] = nlp_u_interpolate(self.t[0], [self.t[1], self.t[2]], [self.eps_c[1], self.eps_c[2]], 2)

    def store_mc(self, scene):
        self.t.append(scene.t)
        self.x.append(scene.x)
        self.y.append(scene.y)
        self.v.append(scene.v)
        self.a.append(scene.a)
        self.eps.append(scene.eps)
        self.a_c.append(scene.a_c)
        self.eps_c.append(scene.eps_c)

    def save(self, save_name):
        t = np.array(self.t, dtype=object)
        x = np.array(self.x, dtype=object)
        y = np.array(self.y, dtype=object)
        v = np.array(self.v, dtype=object)
        a = np.array(self.a, dtype=object)
        eps = np.array(self.eps, dtype=object)
        a_c = np.array(self.a_c, dtype=object)
        eps_c = np.array(self.eps_c, dtype=object)
        np.savez(f'Save/{save_name}', t=t, x=x, y=y, v=v, a=a, eps=eps, a_c=a_c, eps_c=eps_c)

    def load(self, save_name):
        data = np.load(f'Save/{save_name}.npz', allow_pickle=True)
        self.t = data['t']
        self.x = data['x']
        self.y = data['y']
        self.v = data['v']
        self.a = data['a']
        self.eps = data['eps']
        self.a_c = data['a_c']
        self.eps_c = data['eps_c']


class SaveIteration:

    def __init__(self):
        self.reset()

    def reset(self):
        # UAV
        self.traj_p = SaveUAV()
        self.theta_p = []
        self.V_p = []
        self.dV_p = []
        self.ds_p = []
        self.ds_type_p = []
        self.B_p = []
        self.delta_p = []
        self.rou_p = []
        # vessel
        self.traj_s = SaveUGV()
        self.theta_s = []
        self.V_s = []
        self.dV_s = []
        self.ds_s = []
        self.ds_type_s = []
        self.B_s = []
        self.delta_s = []
        self.rou_s = []

    def store_info(self, traj, TR, vehicle_num):
        if vehicle_num == 1:
            self.traj_p.store_mc(traj)
            self.theta_p.append(TR.theta_prev)
            self.V_p.append(TR.V)
            self.dV_p.append(TR.g)
            self.ds_p.append(TR.p)
            self.ds_type_p.append(TR.p_type)
            self.B_p.append(TR.B)
            self.delta_p.append(TR.delta_prev)
            self.rou_p.append(TR.rho)
        else:
            self.traj_s.store_mc(traj)
            self.theta_s.append(TR.theta_prev)
            self.V_s.append(TR.V)
            self.dV_s.append(TR.g)
            self.ds_s.append(TR.p)
            self.ds_type_s.append(TR.p_type)
            self.B_s.append(TR.B)
            self.delta_s.append(TR.delta_prev)
            self.rou_s.append(TR.rho)

    def save(self, save_name):
        # UAV
        traj_p = np.array(self.traj_p, dtype=object)
        theta_p = np.array(self.theta_p, dtype=object)
        V_p = np.array(self.V_p, dtype=object)
        dV_p = np.array(self.dV_p, dtype=object)
        ds_p = np.array(self.ds_p, dtype=object)
        ds_type_p = np.array(self.ds_type_p, dtype=object)
        B_p = np.array(self.B_p, dtype=object)
        delta_p = np.array(self.delta_p, dtype=object)
        rou_p = np.array(self.rou_p, dtype=object)
        # UGV
        traj_s = np.array(self.traj_s, dtype=object)
        theta_s = np.array(self.theta_s, dtype=object)
        V_s = np.array(self.V_s, dtype=object)
        dV_s = np.array(self.dV_s, dtype=object)
        ds_s = np.array(self.ds_s, dtype=object)
        ds_type_s = np.array(self.ds_type_s, dtype=object)
        B_s = np.array(self.B_s, dtype=object)
        delta_s = np.array(self.delta_s, dtype=object)
        rou_s = np.array(self.rou_s, dtype=object)
        np.savez(f'Save/{save_name}',
                 traj_p=traj_p, theta_p=theta_p, V_p=V_p,dV_p=dV_p,
                 ds_p=ds_p, ds_type_p=ds_type_p, B_p=B_p, delta_p=delta_p, rou_p=rou_p,
                 traj_s=traj_s, theta_s=theta_s, V_s=V_s, dV_s=dV_s,
                 ds_s=ds_s, ds_type_s=ds_type_s, B_s=B_s, delta_s=delta_s, rou_s=rou_s)

    def load(self, save_name):
        data = np.load(f'Save/{save_name}.npz', allow_pickle=True)
        # parafoil
        self.traj_p = data['traj_p']
        self.theta_p = data['theta_p']
        self.V_p = data['V_p']
        self.dV_p = data['dV_p']
        self.ds_p = data['ds_p']
        self.ds_type_p = data['ds_type_p']
        self.B_p = data['B_p']
        self.delta_p = data['delta_p']
        self.rou_p = data['rou_p']
        # vessel
        self.traj_s = data['traj_s']
        self.theta_s = data['theta_s']
        self.V_s = data['V_s']
        self.dV_s = data['dV_s']
        self.ds_s = data['ds_s']
        self.ds_type_s = data['ds_type_s']
        self.B_s = data['B_s']
        self.delta_s = data['delta_s']
        self.rou_s = data['rou_s']
