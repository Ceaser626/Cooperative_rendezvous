import numpy as np
from pyomo.environ import *
from pyomo.dae import *
from Utils.tools import interpolate
from Env.save_tools import SaveUAV


def uav_control_model(t_i, state, p_transfer, ref_traj, init_traj, nfe_num):
    # parameter
    g = 9.81
    rou = 1.225
    mass = 1.56
    s = 0.2589
    C_D0 = 0.1631
    K = 0.04525
    # simulation info
    x_f, y_f, psi_f, t_f = p_transfer
    frequency_guide = 0.1  # 0.1Hz
    frequency_control = 1  # 1Hz
    predict_time = 1 / frequency_guide
    control_time = 1 / frequency_control

    m = ConcreteModel(name='Decentralize control (UAV)')

    m.x_f = Param(initialize=np.clip(x_f, -20, 5000), mutable=True)
    m.y_f = Param(initialize=np.clip(y_f, -800, 800), mutable=True)
    m.psi_f = Param(initialize=np.clip(psi_f, -np.pi/2, np.pi/2), mutable=True)
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

    m.x = Var(m.tau, bounds=(-20, 5000))
    m.y = Var(m.tau, bounds=(-800, 800))
    m.h = Var(m.tau, bounds=(0, 400))
    m.v = Var(m.tau, bounds=(12, 20))
    m.eps = Var(m.tau, bounds=(np.radians(-90), np.radians(90)))
    m.gamma = Var(m.tau, bounds=(np.radians(-50), np.radians(50)))
    m.T = Var(m.tau, bounds=(0, 2))
    m.phi = Var(m.tau, bounds=(np.radians(-24), np.radians(24)))
    m.CL = Var(m.tau, bounds=(-0.7, 0.7))
    m.dt_x = DerivativeVar(m.x)
    m.dt_y = DerivativeVar(m.y)
    m.dt_h = DerivativeVar(m.h)
    m.dt_v = DerivativeVar(m.v)
    m.dt_eps = DerivativeVar(m.eps)
    m.dt_gamma = DerivativeVar(m.gamma)

    discretizer = TransformationFactory('dae.collocation')
    discretizer.apply_to(m, wrt=m.tau, nfe=nfe_num, ncp=3, scheme='LAGRANGE-RADAU')
    discretizer.reduce_collocation_points(m, var=m.T, ncp=1, contset=m.tau)
    discretizer.reduce_collocation_points(m, var=m.phi, ncp=1, contset=m.tau)
    discretizer.reduce_collocation_points(m, var=m.CL, ncp=1, contset=m.tau)

    m = initialize_uav_var(m, t_i, init_traj)

    x_ref = []
    y_ref = []
    h_ref = []
    v_ref = []
    eps_ref = []
    gamma_ref = []
    T_ref = []
    phi_ref = []
    CL_ref = []
    f_x = interpolate(ref_traj.t, ref_traj.x)
    f_y = interpolate(ref_traj.t, ref_traj.y)
    f_h = interpolate(ref_traj.t, ref_traj.h)
    f_v = interpolate(ref_traj.t, ref_traj.v)
    f_eps = interpolate(ref_traj.t, ref_traj.eps)
    f_gamma = interpolate(ref_traj.t, ref_traj.gamma)
    f_T = interpolate(ref_traj.t, ref_traj.T)
    f_phi = interpolate(ref_traj.t, ref_traj.phi)
    f_CL = interpolate(ref_traj.t, ref_traj.CL)

    for i in m.tau:
        inst = t_i + i * m.t_f
        x = np.clip(f_x(inst).item(), -20, 5000)
        y = np.clip(f_y(inst).item(), -800, 800)
        h = np.clip(f_h(inst).item(), 0, 100)
        v = np.clip(f_v(inst).item(), 0, 25)
        eps = np.clip(f_eps(inst).item(), np.radians(-90), np.radians(90))
        gamma = np.clip(f_gamma(inst).item(), np.radians(-50), np.radians(50))
        T = np.clip(f_T(inst).item(), 0, 2)
        phi = np.clip(f_phi(inst).item(), np.radians(-24), np.radians(24))
        CL = np.clip(f_CL(inst).item(), -0.7, 0.7)
        x_ref.append(x)
        y_ref.append(y)
        h_ref.append(h)
        v_ref.append(v)
        eps_ref.append(eps)
        gamma_ref.append(gamma)
        T_ref.append(T)
        phi_ref.append(phi)
        CL_ref.append(CL)

    i, sum_delta_x = 0, 0
    k_x, k_y, k_h = 1, 1, 1
    k_v, k_eps, k_gamma = 1, 1e1, 1e1
    for j in m.tau:
        if j != 0:
            sum_delta_x = sum_delta_x + k_x * (m.x[j] - x_ref[i]) ** 2 \
                                      + k_y * (m.y[j] - y_ref[i]) ** 2 \
                                      + k_h * (m.h[j] - h_ref[i]) ** 2 \
                                      + k_v * (m.v[j] - v_ref[i]) ** 2 \
                                      + k_eps * (m.eps[j] - eps_ref[i]) ** 2 \
                                      + k_gamma * (m.gamma[j] - gamma_ref[i]) ** 2
        i = i + 1
    m.sum_delta_x = sum_delta_x

    i, sum_u = 0, 0
    k_T, k_phi, k_CL = 1, 1, 1
    for j in m.tau:
        if j != 0:
            sum_u = sum_u + k_T * ((m.T[j] - T_ref[i])/2) ** 2 \
                          + k_phi * ((m.phi[j] - phi_ref[i])/np.radians(20)) ** 2 \
                          + k_CL * ((m.CL[j] - CL_ref[i])/0.7) ** 2
        i = i + 1
    m.sum_u = sum_u

    def obj_rule(m):
        return 1e1 * m.sum_delta_x + m.sum_u
    m.obj = Objective(rule=obj_rule, sense=minimize)

    def init_bound_rule(m):
        yield m.time[0] == t_i
        yield m.x[0] == np.clip(state[0], -20, 5000)
        yield m.y[0] == np.clip(state[1], -800, 800)
        yield m.h[0] == np.clip(state[2], 0, 400)
        yield m.v[0] == np.clip(state[3], 12, 20)
        yield m.eps[0] == np.clip(state[4], np.radians(-90), np.radians(90))
        yield m.gamma[0] == np.clip(state[5], np.radians(-50), np.radians(50))
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

    for i in m.tau:
        inst = t_i + i * m.t_f
        x = np.clip(np.nan_to_num(f_x(inst).item()), -20, 5000)
        y = np.clip(np.nan_to_num(f_y(inst).item()), -800, 800)
        h = np.clip(np.nan_to_num(f_h(inst).item()), 0, 400)
        v = np.clip(np.nan_to_num(f_v(inst).item()), 12, 20)
        eps = np.clip(np.nan_to_num(f_eps(inst).item()), np.radians(-90), np.radians(90))
        gamma = np.clip(np.nan_to_num(f_gamma(inst).item()), np.radians(-50), np.radians(50))
        T = np.clip(np.nan_to_num(f_T(inst).item()), 0, 2)
        phi = np.clip(np.nan_to_num(f_phi(inst).item()), np.radians(-24), np.radians(24))
        CL = np.clip(np.nan_to_num(f_CL(inst).item()), -0.7, 0.7)

        m.time[i] = inst
        m.x[i] = x
        m.y[i] = y
        m.h[i] = h
        m.v[i] = v
        m.eps[i] = eps
        m.gamma[i] = gamma
        m.T[i] = T
        m.phi[i] = phi
        m.CL[i] = CL
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

    return m


def linear_init_p(t_c, obs_c, theta):
    x_f, y_f, psi_f, t_f = theta
    traj = SaveUAV()

    t = t_c
    obs = obs_c
    a = [1, 0, 0]
    traj.store_sim(t, obs, a)

    t = t_f
    obs = [x_f, y_f, 0, 16, 0, 0]
    a = [0, 0, 0]
    traj.store_sim(t, obs, a)

    return traj
