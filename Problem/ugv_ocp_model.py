import numpy as np
from pyomo.environ import *
from pyomo.dae import *
from Utils.tools import interpolate
from Env.save_tools import SaveUGV


def decentralized_ugv(t_i, state, ref, p_transfer, nfe_num):
    # parameter
    tao_a = 1.5
    tao_eps = 2
    # simulation info
    x_f, y_f, psi_f, t_f = p_transfer

    m = ConcreteModel(name='Decentralize guidance (UGV)')

    m.x_f = Param(initialize=np.clip(x_f, 0, 5000), mutable=True)
    m.y_f = Param(initialize=np.clip(y_f, -800, 800), mutable=True)
    m.psi_f = Param(initialize=np.clip(psi_f, -np.pi/2, np.pi/2), mutable=True)
    m.tf = Param(initialize=np.clip(t_f, 0.1, 400), mutable=True)

    m.tau = ContinuousSet(bounds=(0, 1))
    m.time = Var(m.tau, bounds=(0 + t_i, 400 + t_i))
    m.t_f = Var(within=NonNegativeReals, bounds=(0.1, 400))
    m.dt_time = DerivativeVar(m.time)

    m.x = Var(m.tau, bounds=(0, 5000))
    m.y = Var(m.tau, bounds=(-800, 800))
    m.v = Var(m.tau, bounds=(0, 30))
    m.a = Var(m.tau, bounds=(-5, 5))
    m.eps = Var(m.tau, bounds=(np.radians(-90), np.radians(90)))
    m.a_c = Var(m.tau, bounds=(-5, 5))
    m.eps_c = Var(m.tau, bounds=(np.radians(-30), np.radians(30)))
    m.sum_u = Var(m.tau, bounds=(0, 1e4))
    m.dt_x = DerivativeVar(m.x)
    m.dt_y = DerivativeVar(m.y)
    m.dt_v = DerivativeVar(m.v)
    m.dt_a = DerivativeVar(m.a)
    m.dt_eps = DerivativeVar(m.eps)
    m.dt_sum_u = DerivativeVar(m.sum_u)

    m.p_x = Var(within=NonNegativeReals, bounds=(0, 1000))
    m.p_y = Var(within=NonNegativeReals, bounds=(0, 1000))
    m.p_eps = Var(within=NonNegativeReals, bounds=(0, 2 * np.pi))
    m.p_v = Var(within=NonNegativeReals, bounds=(0, 20))
    m.p_tf = Var(within=NonNegativeReals, bounds=(0, 700))
    m.u_x = Var(within=NonNegativeReals, bounds=(0, 1000))
    m.u_y = Var(within=NonNegativeReals, bounds=(0, 1000))
    m.u_eps = Var(within=NonNegativeReals, bounds=(0, 2 * np.pi))
    m.u_v = Var(within=NonNegativeReals, bounds=(0, 20))
    m.u_tf = Var(within=NonNegativeReals, bounds=(0, 700))

    discretizer = TransformationFactory('dae.collocation')
    discretizer.apply_to(m, wrt=m.tau, nfe=nfe_num, ncp=3, scheme='LAGRANGE-RADAU')

    m.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
    m.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
    m.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
    m.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)
    m.dual = Suffix(direction=Suffix.IMPORT_EXPORT)

    m = initialize_ugv_var(m, t_i, ref)

    m.V = (m.x[1] - m.x_f) ** 2 \
        + (m.y[1] - m.y_f) ** 2 \
        + m.sum_u[1]

    beta = 1e1
    m.L1_penalty = beta * (m.p_x + m.u_x +\
                           m.p_y + m.u_y +\
                           m.p_eps + m.u_eps +\
                           m.p_v + m.u_v +\
                           m.p_tf + m.u_tf)

    def obj_rule(m):
        return m.sum_u[1] + m.L1_penalty
    m.obj = Objective(rule=obj_rule, sense=minimize)

    def init_bound_rule(m):
        yield m.time[0] == t_i
        yield m.x[0] == np.clip(state[0], -200, 5000)
        yield m.y[0] == np.clip(state[1], -800, 800)
        yield m.v[0] == np.clip(state[2], 0, 30)
        yield m.a[0] == np.clip(state[3], -5, 5)
        yield m.eps[0] == np.clip(state[4], np.radians(-90), np.radians(90))
        yield m.sum_u[0] == 0
    m.init_bound = ConstraintList(rule=init_bound_rule)

    def terminal_x_rule(m):
        return m.x[1] - m.x_f == m.p_x - m.u_x
    m.terminal_x = Constraint(rule=terminal_x_rule)

    def terminal_y_rule(m):
        return m.y[1] - m.y_f == m.p_y - m.u_y
    m.terminal_y = Constraint(rule=terminal_y_rule)

    def terminal_eps_rule(m):
        return m.eps[1] - m.psi_f == m.p_eps - m.u_eps
    m.terminal_eps = Constraint(rule=terminal_eps_rule)

    def terminal_v_rule(m):
        return m.v[1] - 12 == m.p_v - m.u_v
    m.terminal_v = Constraint(rule=terminal_v_rule)

    def terminal_tf_rule(m):
        return m.time[1] - m.tf == m.p_tf - m.u_tf
    m.terminal_tf = Constraint(rule=terminal_tf_rule)

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
            return m.dt_x[i] == m.t_f * (m.v[i] * cos(m.eps[i]))
    m.ode_x = Constraint(m.tau, rule=ode_x_rule)

    def ode_y_rule(m, i):
        if i == 0:
            return Constraint.Skip
        else:
            return m.dt_y[i] == m.t_f * (m.v[i] * sin(m.eps[i]))
    m.ode_y = Constraint(m.tau, rule=ode_y_rule)

    def ode_v_rule(m, i):
        if i == 0:
            return Constraint.Skip
        else:
            return m.dt_v[i] == m.t_f * m.a[i]
    m.ode_v = Constraint(m.tau, rule=ode_v_rule)

    def ode_a_rule(m, i):
        if i == 0:
            return Constraint.Skip
        else:
            return m.dt_a[i] == m.t_f * (m.a_c[i] - m.a[i]) / tao_a
    m.ode_a = Constraint(m.tau, rule=ode_a_rule)

    def ode_eps_rule(m, i):
        if i == 0:
            return Constraint.Skip
        else:
            return m.dt_eps[i] == m.t_f * (m.eps_c[i] - m.eps[i]) / tao_eps
    m.ode_eps = Constraint(m.tau, rule=ode_eps_rule)

    def ode_sum_u_rule(m, i):
        if i == 0:
            return Constraint.Skip
        else:
            return m.dt_sum_u[i] == m.t_f * ((m.a_c[i]/10) ** 2 + (m.eps_c[i]/np.radians(10)) ** 2)
    m.ode_sum_u = Constraint(m.tau, rule=ode_sum_u_rule)

    paramList = [m.x_f, m.y_f, m.psi_f, m.tf]
    m.paramList = paramList


    return m, paramList


def initialize_ugv_var(m, t_i, ref):
    t = ref.t
    x = ref.x
    y = ref.y
    v = ref.v
    a = ref.a
    eps = ref.eps
    a_c = ref.a_c
    eps_c = ref.eps_c

    f_x = interpolate(t, x)
    f_y = interpolate(t, y)
    f_v = interpolate(t, v)
    f_a = interpolate(t, a)
    f_eps = interpolate(t, eps)
    f_a_c = interpolate(t, a_c)
    f_eps_c = interpolate(t, eps_c)

    m.t_f = t[-1]
    for i in m.tau:
        inst = i * t[-1]
        m.time[i] = inst + t_i
        x = np.clip(np.nan_to_num(f_x(inst).item()), 0, 5000)
        y = np.clip(np.nan_to_num(f_y(inst).item()), -800, 800)
        v = np.clip(np.nan_to_num(f_v(inst).item()), 0, 30)
        a = np.clip(np.nan_to_num(f_a(inst).item()), -5, 5)
        eps = np.clip(np.nan_to_num(f_eps(inst).item()), np.radians(-90), np.radians(90))
        a_c = np.clip(np.nan_to_num(f_a_c(inst).item()), -5, 5)
        eps_c = np.clip(np.nan_to_num(f_eps_c(inst).item()), np.radians(-30), np.radians(30))

        m.x[i] = x
        m.y[i] = y
        m.v[i] = v
        m.a[i] = a
        m.eps[i] = eps
        m.a_c[i] = a_c
        m.eps_c[i] = eps_c

        m.dt_time[i] = t[-1]
        tao_a = 1.5
        tao_eps = 2
        m.dt_x[i] = (v * cos(eps)) * t[-1]
        m.dt_y[i] = (v * sin(eps)) * t[-1]
        m.dt_v[i] = a * t[-1]
        m.dt_a[i] = (a_c - a) / tao_a * t[-1]
        m.dt_a[i] = (eps_c - eps) / tao_eps * t[-1]
        m.dt_sum_u[i] = ((a_c/10) ** 2 + (eps_c/np.radians(10)) ** 2) * t[-1]

    return m


def linear_init_ugv(t_g, obs_g, theta):
    x_f, y_f, psi_f, t_f = theta
    traj = SaveUGV()

    t = t_g
    obs = obs_g
    a = [0, 0]
    traj.store_sim(t, obs, a)

    t = t_f
    obs = [x_f, y_f, 16, 0, 0]
    a = [0, 0]
    traj.store_sim(t, obs, a)

    return traj
