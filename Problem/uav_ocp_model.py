import numpy as np
from pyomo.environ import *
from pyomo.dae import *
from Utils.tools import interpolate
from Env.save_tools import SaveUAV


def decentralized_uav(t_i, state, ref, p_transfer, nfe_num):
    # parameter
    g = 9.81
    rou = 1.225
    mass = 1.56
    s = 0.2589
    C_D0 = 0.1631
    K = 0.04525
    # simulation info
    x_f, y_f, psi_f, t_f = p_transfer

    m = ConcreteModel(name='Decentralize guidance (UAV)')

    m.x_f = Param(initialize=np.clip(x_f, -20, 5000), mutable=True)
    m.y_f = Param(initialize=np.clip(y_f, -800, 800), mutable=True)
    m.psi_f = Param(initialize=np.clip(psi_f, -np.pi/2, np.pi/2), mutable=True)
    m.tf = Param(initialize=np.clip(t_f, 0.1, 400), mutable=True)

    m.tau = ContinuousSet(bounds=(0, 1))
    m.time = Var(m.tau, bounds=(0 + t_i, 400 + t_i))
    m.t_f = Var(within=NonNegativeReals, bounds=(0.1, 400))
    m.dt_time = DerivativeVar(m.time)

    m.x = Var(m.tau, bounds=(-20, 5000))
    m.y = Var(m.tau, bounds=(-800, 800))
    m.h = Var(m.tau, bounds=(0, 400))
    m.v = Var(m.tau, bounds=(12, 20))
    m.eps = Var(m.tau, bounds=(np.radians(-90), np.radians(90)))
    m.gamma = Var(m.tau, bounds=(np.radians(-50), np.radians(50)))
    m.T = Var(m.tau, bounds=(0, 2))
    m.phi = Var(m.tau, bounds=(np.radians(-24), np.radians(24)))
    m.CL = Var(m.tau, bounds=(-0.7, 0.7))
    m.dT = Var(m.tau, bounds=(-1, 1))
    m.dphi = Var(m.tau, bounds=(np.radians(-10), np.radians(10)))
    m.dCL = Var(m.tau, bounds=(-0.2, 0.2))
    m.sum_u = Var(m.tau, bounds=(0, 1e8))
    m.dt_x = DerivativeVar(m.x)
    m.dt_y = DerivativeVar(m.y)
    m.dt_h = DerivativeVar(m.h)
    m.dt_v = DerivativeVar(m.v)
    m.dt_eps = DerivativeVar(m.eps)
    m.dt_gamma = DerivativeVar(m.gamma)
    m.dt_T = DerivativeVar(m.T)
    m.dt_phi = DerivativeVar(m.phi)
    m.dt_CL = DerivativeVar(m.CL)
    m.dt_sum_u = DerivativeVar(m.sum_u)

    m.p_x = Var(within=NonNegativeReals, bounds=(0, 1000))
    m.p_y = Var(within=NonNegativeReals, bounds=(0, 1000))
    m.p_h = Var(within=NonNegativeReals, bounds=(0, 1000))
    m.p_eps = Var(within=NonNegativeReals, bounds=(0, 2 * np.pi))
    m.p_tf = Var(within=NonNegativeReals, bounds=(0, 700))
    m.u_x = Var(within=NonNegativeReals, bounds=(0, 1000))
    m.u_y = Var(within=NonNegativeReals, bounds=(0, 1000))
    m.u_h = Var(within=NonNegativeReals, bounds=(0, 1000))
    m.u_eps = Var(within=NonNegativeReals, bounds=(0, 2 * np.pi))
    m.u_tf = Var(within=NonNegativeReals, bounds=(0, 700))

    discretizer = TransformationFactory('dae.collocation')
    discretizer.apply_to(m, wrt=m.tau, nfe=nfe_num, ncp=3, scheme='LAGRANGE-RADAU')

    m.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
    m.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
    m.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
    m.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)
    m.dual = Suffix(direction=Suffix.IMPORT_EXPORT)

    m = initialize_uav_var(m, t_i, ref)

    m.V = (m.x[1] - m.x_f) ** 2 \
        + (m.y[1] - m.y_f) ** 2 \
        + m.sum_u[1]

    beta = 1e1
    m.L1_penalty = beta * (m.p_x + m.u_x +\
                           m.p_y + m.u_y +\
                           m.p_h + m.u_h +\
                           m.p_eps + m.u_eps +\
                           m.p_tf + m.u_tf)

    def obj_rule(m):
        return m.sum_u[1] + m.L1_penalty
    m.obj = Objective(rule=obj_rule, sense=minimize)

    def init_bound_rule(m):
        yield m.time[0] == t_i
        yield m.x[0] == np.clip(state[0], -20, 5000)
        yield m.y[0] == np.clip(state[1], -800, 800)
        yield m.h[0] == np.clip(state[2], 0, 100)
        yield m.v[0] == np.clip(state[3], 0, 25)
        yield m.eps[0] == np.clip(state[4], np.radians(-90), np.radians(90))
        yield m.gamma[0] == np.clip(state[5], np.radians(-50), np.radians(50))
        yield m.sum_u[0] == 0
    m.init_bound = ConstraintList(rule=init_bound_rule)

    def terminal_x_rule(m):
        return m.x[1] - m.x_f == m.p_x - m.u_x
    m.terminal_x = Constraint(rule=terminal_x_rule)

    def terminal_y_rule(m):
        return m.y[1] - m.y_f == m.p_y - m.u_y
    m.terminal_y = Constraint(rule=terminal_y_rule)

    def terminal_h_rule(m):
        return m.h[1] == m.p_h - m.u_h
    m.terminal_h = Constraint(rule=terminal_h_rule)

    def terminal_eps_rule(m):
        return m.eps[1] - m.psi_f == m.p_eps - m.u_eps
    m.terminal_eps = Constraint(rule=terminal_eps_rule)

    def terminal_tf_rule(m):
        return m.time[1] - m.tf == m.p_tf - m.u_tf
    m.terminal_tf = Constraint(rule=terminal_tf_rule)

    def path_v_rule(m, i):
        return inequality(12, m.v[i], 20)
    m.path_v = Constraint(m.tau, rule=path_v_rule)

    def ode_t_rule(m, i):
        if i == 0:
            return Constraint.Skip
        else:
            return m.dt_time[i] == m.t_f
    m.ode_t = Constraint(m.tau, rule=ode_t_rule)

    def ode_x_rule(m, i):
        if i == 0:
            return Constraint.Skip
        else:
            return m.dt_x[i] == m.t_f * (m.v[i]*cos(m.gamma[i])*cos(m.eps[i]))
    m.ode_x = Constraint(m.tau, rule=ode_x_rule)

    def ode_y_rule(m, i):
        if i == 0:
            return Constraint.Skip
        else:
            return m.dt_y[i] == m.t_f * (m.v[i]*cos(m.gamma[i])*sin(m.eps[i]))
    m.ode_y = Constraint(m.tau, rule=ode_y_rule)

    def ode_h_rule(m, i):
        if i == 0:
            return Constraint.Skip
        else:
            return m.dt_h[i] == m.t_f * (m.v[i]*sin(m.gamma[i]))
    m.ode_h = Constraint(m.tau, rule=ode_h_rule)

    def ode_v_rule(m, i):
        if i == 0:
            return Constraint.Skip
        else:
            C_D = C_D0 + K * m.CL[i]**2
            D = 0.5 * rou * m.v[i] ** 2 * s * C_D
            return m.dt_v[i] == m.t_f * ((m.T[i] - D)/mass - g * sin(m.gamma[i]))
    m.ode_v = Constraint(m.tau, rule=ode_v_rule)

    def ode_eps_rule(m, i):
        if i == 0:
            return Constraint.Skip
        else:
            L = 0.5 * rou * m.v[i] ** 2 * s * m.CL[i]
            return m.dt_eps[i] == m.t_f * (L * sin(m.phi[i]) / (mass * m.v[i] * cos(m.gamma[i])))
    m.ode_eps = Constraint(m.tau, rule=ode_eps_rule)

    def ode_gamma_rule(m, i):
        if i == 0:
            return Constraint.Skip
        else:
            L = 0.5 * rou * m.v[i] ** 2 * s * m.CL[i]
            return m.dt_gamma[i] == m.t_f * (L * cos(m.phi[i]) / (mass * m.v[i]) - g * cos(m.gamma[i]) / m.v[i])
    m.ode_gamma = Constraint(m.tau, rule=ode_gamma_rule)

    def ode_T_rule(m, i):
        if i == 0:
            return Constraint.Skip
        else:
            return m.dt_T[i] == m.t_f * m.dT[i]
    m.ode_T = Constraint(m.tau, rule=ode_T_rule)

    def ode_phi_rule(m, i):
        if i == 0:
            return Constraint.Skip
        else:
            return m.dt_phi[i] == m.t_f * m.dphi[i]
    m.ode_phi = Constraint(m.tau, rule=ode_phi_rule)

    def ode_CL_rule(m, i):
        if i == 0:
            return Constraint.Skip
        else:
            return m.dt_CL[i] == m.t_f * m.dCL[i]
    m.ode_CL = Constraint(m.tau, rule=ode_CL_rule)

    def ode_sum_u_rule(m, i):
        if i == 0:
            return Constraint.Skip
        else:
            return m.dt_sum_u[i] == m.t_f * ((m.T[i]/5) ** 2 + (m.phi[i]/np.radians(20)) ** 2 + (m.CL[i]/0.7) ** 2)
    m.ode_sum_u = Constraint(m.tau, rule=ode_sum_u_rule)

    paramList = [m.x_f, m.y_f, m.psi_f, m.tf]
    m.paramList = paramList

    return m, paramList


def initialize_uav_var(m, t_i, ref):
    t = ref.t
    x = ref.x
    y = ref.y
    h = ref.h
    v = ref.v
    eps = ref.eps
    gamma = ref.gamma
    T = ref.T
    phi = ref.phi
    CL = ref.CL

    f_x = interpolate(t, x)
    f_y = interpolate(t, y)
    f_h = interpolate(t, h)
    f_v = interpolate(t, v)
    f_eps = interpolate(t, eps)
    f_gamma = interpolate(t, gamma)
    f_T = interpolate(t, T)
    f_phi = interpolate(t, phi)
    f_CL = interpolate(t, CL)

    m.t_f = t[-1]
    for i in m.tau:
        inst = i * t[-1]
        m.time[i] = inst + t_i
        x = np.clip(np.nan_to_num(f_x(inst).item()), -20, 5000)
        y = np.clip(np.nan_to_num(f_y(inst).item()), -800, 800)
        h = np.clip(np.nan_to_num(f_h(inst).item()), 0, 400)
        v = np.clip(np.nan_to_num(f_v(inst).item()), 12, 20)
        eps = np.clip(np.nan_to_num(f_eps(inst).item()), np.radians(-90), np.radians(90))
        gamma = np.clip(np.nan_to_num(f_gamma(inst).item()), np.radians(-50), np.radians(50))
        T = np.clip(np.nan_to_num(f_T(inst).item()), 0, 2)
        phi = np.clip(np.nan_to_num(f_phi(inst).item()), np.radians(-24), np.radians(24))
        CL = np.clip(np.nan_to_num(f_CL(inst).item()), -0.7, 0.7)

        m.x[i] = x
        m.y[i] = y
        m.h[i] = h
        m.v[i] = v
        m.eps[i] = eps
        m.gamma[i] = gamma
        m.T[i] = T
        m.phi[i] = phi
        m.CL[i] = CL

        m.dt_time[i] = t[-1]

        g = 9.81
        rou = 1.225
        mass = 1.56
        s = 0.2589
        C_D0 = 0.1631
        K = 0.04525

        C_D = C_D0 + K * CL ** 2
        L = 0.5 * rou * v ** 2 * s * CL
        D = 0.5 * rou * v ** 2 * s * C_D
        m.dt_x[i] = (v * cos(gamma) * cos(eps)) * t[-1]
        m.dt_y[i] = (v * cos(gamma) * sin(eps)) * t[-1]
        m.dt_h[i] = (v * sin(gamma)) * t[-1]
        m.dt_v[i] = ((T - D) / mass - g * sin(gamma)) * t[-1]
        m.dt_eps[i] = (L * sin(phi) / (mass * v * cos(gamma))) * t[-1]
        m.dt_gamma[i] = (L * cos(phi) / (mass * v) - g * cos(gamma) / v) * t[-1]
        m.dt_sum_u[i] = ((T/2) ** 2 + (phi/np.radians(20)) ** 2 + (CL/0.7) ** 2) * t[-1]

    return m


def linear_init_uav(t_g, obs_g, theta):
    x_f, y_f, psi_f, t_f = theta
    traj = SaveUAV()

    t = t_g
    obs = obs_g
    a = [1, 0, 0]
    traj.store_sim(t, obs, a)

    t = t_f
    obs = [x_f, y_f, 0, 16, 0, 0]
    a = [0, 0, 0]
    traj.store_sim(t, obs, a)

    return traj
