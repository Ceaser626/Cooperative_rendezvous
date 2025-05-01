import numpy as np
from pyomo.environ import *
from pyomo.dae import *
from Utils.tools import interpolate
from Env.save_tools import SaveUGV


def ugv_control_model(t_i, state, p_transfer, ref_traj, init_traj, nfe_num):
    # parameter
    tao_a = 1.5
    tao_eps = 2
    # simulation info
    x_f, t_f, y_f, psi_f = p_transfer
    frequency_guide = 0.1  # 0.1Hz
    frequency_control = 1  # 1Hz
    predict_time = 1 / frequency_guide
    control_time = 1 / frequency_control

    m = ConcreteModel(name='Decentralize control (AUV)')

    m.x_f = Param(initialize=np.clip(x_f, 0, 5000), mutable=True)
    m.y_f = Param(initialize=np.clip(y_f, -1000, 1000), mutable=True)
    m.psi_f = Param(initialize=np.clip(psi_f, -np.pi / 2, np.pi / 2), mutable=True)
    m.tf = Param(initialize=np.clip(t_f, 0.1, 400), mutable=True)

    t_eta = t_f - t_i
    if control_time <= t_eta <= predict_time:
        t_len = int(np.ceil(t_eta / control_time)) * control_time
        nfe_num = int(np.ceil(t_eta / control_time))
    elif t_eta <= control_time:
        t_len = control_time
        nfe_num = 1
    else:
        t_len = predict_time
    m.tau = ContinuousSet(bounds=(0, 1))
    m.time = Var(m.tau, bounds=(t_i, t_len + t_i))
    m.t_f = Param(initialize=t_len)
    m.dt_time = DerivativeVar(m.time)

    m.x = Var(m.tau, bounds=(0, 5000))
    m.y = Var(m.tau, bounds=(-800, 800))
    m.v = Var(m.tau, bounds=(0, 30))
    m.a = Var(m.tau, bounds=(-5, 5))
    m.eps = Var(m.tau, bounds=(np.radians(-90), np.radians(90)))
    m.a_c = Var(m.tau, bounds=(-5, 5))
    m.eps_c = Var(m.tau, bounds=(np.radians(-30), np.radians(30)))
    m.dt_x = DerivativeVar(m.x)
    m.dt_y = DerivativeVar(m.y)
    m.dt_v = DerivativeVar(m.v)
    m.dt_a = DerivativeVar(m.a)
    m.dt_eps = DerivativeVar(m.eps)

    discretizer = TransformationFactory('dae.collocation')
    discretizer.apply_to(m, wrt=m.tau, nfe=nfe_num, ncp=3, scheme='LAGRANGE-RADAU')
    discretizer.reduce_collocation_points(m, var=m.a_c, ncp=1, contset=m.tau)
    discretizer.reduce_collocation_points(m, var=m.eps_c, ncp=1, contset=m.tau)

    m = initialize_ugv_var(m, t_i, init_traj)

    x_ref = []
    y_ref = []
    v_ref = []
    a_ref = []
    eps_ref = []
    a_c_ref = []
    eps_c_ref = []
    f_x = interpolate(ref_traj.t, ref_traj.x)
    f_y = interpolate(ref_traj.t, ref_traj.y)
    f_v = interpolate(ref_traj.t, ref_traj.v)
    f_a = interpolate(ref_traj.t, ref_traj.a)
    f_eps = interpolate(ref_traj.t, ref_traj.eps)
    f_a_c = interpolate(ref_traj.t, ref_traj.a_c)
    f_eps_c = interpolate(ref_traj.t, ref_traj.eps_c)

    for i in m.tau:
        inst = t_i + i * m.t_f
        x = np.clip(f_x(inst).item(), 0, 5000)
        y = np.clip(f_y(inst).item(), -800, 800)
        v = np.clip(f_v(inst).item(), 0, 30)
        a = np.clip(f_a(inst).item(), -5, 5)
        eps = np.clip(f_eps(inst).item(), np.radians(-90), np.radians(90))
        a_c = np.clip(f_a_c(inst).item(), -5, 5)
        eps_c = np.clip(f_eps_c(inst).item(), np.radians(-30), np.radians(30))
        x_ref.append(x)
        y_ref.append(y)
        v_ref.append(v)
        a_ref.append(a)
        eps_ref.append(eps)
        a_c_ref.append(a_c)
        eps_c_ref.append(eps_c)

    i, sum_delta_x = 0, 0
    k_x, k_y, k_v = 5, 5, 1
    k_a, k_eps = 1, 1e1
    for j in m.tau:
        if j != 0:
            sum_delta_x = sum_delta_x + k_x * (m.x[j] - x_ref[i]) ** 2 \
                                      + k_y * (m.y[j] - y_ref[i]) ** 2 \
                                      + k_v * (m.v[j] - v_ref[i]) ** 2 \
                                      + k_a * (m.a[j] - a_ref[i]) ** 2 \
                                      + k_eps * (m.eps[j] - eps_ref[i]) ** 2
        i = i + 1
    m.sum_delta_x = sum_delta_x

    i, sum_u = 0, 0
    k_a_c, k_eps_c = 1, 1
    for j in m.tau:
        if j != 0:
            sum_u = sum_u + k_a_c * ((m.a_c[j] - a_c_ref[i])/10) ** 2 \
                          + k_eps_c * ((m.eps_c[j] - eps_c_ref[i])/np.radians(10)) ** 2
        i = i + 1
    m.sum_u = sum_u

    def obj_rule(m):
        return 1e1 * m.sum_delta_x + m.sum_u
    m.obj = Objective(rule=obj_rule, sense=minimize)

    def init_bound_rule(m):
        yield m.time[0] == t_i
        yield m.x[0] == np.clip(state[0], -200, 5000)
        yield m.y[0] == np.clip(state[1], -800, 800)
        yield m.v[0] == np.clip(state[2], 0, 30)
        yield m.a[0] == np.clip(state[3], -5, 5)
        yield m.eps[0] == np.clip(state[4], np.radians(-90), np.radians(90))
    m.init_bound = ConstraintList(rule=init_bound_rule)

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

    for i in m.tau:
        inst = t_i + i * m.t_f
        x = np.clip(np.nan_to_num(f_x(inst).item()), 0, 5000)
        y = np.clip(np.nan_to_num(f_y(inst).item()), -800, 800)
        v = np.clip(np.nan_to_num(f_v(inst).item()), 0, 30)
        a = np.clip(np.nan_to_num(f_a(inst).item()), -5, 5)
        eps = np.clip(np.nan_to_num(f_eps(inst).item()), np.radians(-90), np.radians(90))
        a_c = np.clip(np.nan_to_num(f_a_c(inst).item()), -5, 5)
        eps_c = np.clip(np.nan_to_num(f_eps_c(inst).item()), np.radians(-30), np.radians(30))

        m.time[i] = inst
        m.x[i] = x
        m.y[i] = y
        m.v[i] = v
        m.a[i] = a
        m.eps[i] = eps
        m.a_c[i] = a_c
        m.eps_c[i] = eps_c
        tao_a = 1.5
        tao_eps = 2
        m.dt_time[i] = t[-1]
        m.dt_x[i] = (v * cos(eps)) * t[-1]
        m.dt_y[i] = (v * sin(eps)) * t[-1]
        m.dt_v[i] = a * t[-1]
        m.dt_a[i] = (a_c - a) / tao_a * t[-1]
        m.dt_a[i] = (eps_c - eps) / tao_eps * t[-1]

    return m


def linear_init_s(t_c, obs_c, theta):
    x_f, t_f, y_f, psi_f = theta
    traj = SaveUGV()

    t = t_c
    obs = obs_c
    a = [0, 0]
    traj.store_sim(t, obs, a)

    t = t_f
    obs = [x_f, y_f, 15, 0, 0]
    a = [0, 0]
    traj.store_sim(t, obs, a)

    return traj
