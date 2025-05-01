import numpy as np


class TrustRegionOpt:

    def __init__(self, para_dict):
        self.iter_num = 0
        self.w = 0.5
        # Local information
        self.theta_prev = np.zeros(2)
        self.V = 0
        self.V_prev = 10
        self.g = np.zeros((2, 1))
        self.g_prev = 0
        self.B = np.eye(2)
        # Trust region parameter
        self.delta = para_dict['delta_init']
        self.delta_prev = self.delta
        self.delta_max = para_dict['delta_max']
        self.rho = 0
        self.eta = para_dict['eta']
        # step
        self.p = np.zeros((2, 1))
        self.p_type = 'none'

    def update_info(self, theta_1, V_1, g_1, B_1, theta_2, V_2, g_2, B_2):
        g_1_T = np.transpose(g_1)
        g_2_T = np.transpose(g_2)
        p_bar = theta_2 - theta_1
        p_bar_T = np.transpose(p_bar)

        self.V = abs(self.w * V_1 + (1 - self.w) * (V_2 - g_2_T @ p_bar + 0.5 * p_bar_T @ B_2 @ p_bar))
        self.g = np.transpose(self.w * g_1_T + (1 - self.w) * (g_2_T - p_bar_T @ B_2))
        self.B = self.w * B_1 + (1 - self.w) * B_2

    def m_func(self, p):
        g_T = np.transpose(self.g)

        m_value = self.V + g_T @ p + 0.5 * np.transpose(p) @ self.B @ p
        return m_value

    def determine_p(self):
        g_T = np.transpose(self.g)

        p_B = - np.linalg.inv(self.B) @ self.g
        p_U = - (g_T @ self.g) / (g_T @ self.B @ self.g) * self.g

        # decide gradient step
        if np.linalg.norm(p_B) <= self.delta:
            p = p_B
            p_type = 'p_B'
            print(f'Perform p_B: {p}')
        elif np.linalg.norm(p_U) >= self.delta:
            p = self.delta * p_U / np.linalg.norm(p_U)
            p_type = 'p_U'
            print(f'Perform bounded p_U: {p}')
        else:
            a = np.transpose(p_B - p_U) @ (p_B - p_U)
            b = 2 * np.transpose(p_U) @ (p_B - p_U)
            c = np.transpose(p_U) @ p_U - self.delta ** 2
            right = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
            left = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
            if b ** 2 - 4 * a * c < 0:
                print('cannot solve due to b**2-4*a*c < 0')
            elif right < 0 or right > 1:
                tau = 1 + left
            else:
                tau = 1 + right
            p = p_U + (tau - 1) * (p_B - p_U)
            p_type = 'p_dogleg'
            print(f'Perform dogleg p: {p}')

        self.p = p
        self.p_type = p_type

    def update_region_size(self, theta, V_part1, theta_2, V_2, g_2, B_2):
        m_0 = self.V
        m_p = self.m_func(self.p)

        p_bar = theta_2 - theta
        p_bar_T = np.transpose(p_bar)
        g_2_T = np.transpose(g_2)
        V_bar = V_2 - g_2_T @ p_bar + 0.5 * p_bar_T @ B_2 @ p_bar
        g_bar_T = g_2_T - p_bar_T @ B_2
        B_bar = B_2
        V_part2 = V_bar + g_bar_T @ self.p + 0.5 * np.transpose(self.p) @ B_bar @ self.p
        V_p = self.w * V_part1 + (1 - self.w) * V_part2

        try:
            self.rho = (m_0 - V_p) / (m_0 - m_p)
        except ZeroDivisionError:
            self.rho = self.eta + 1

        self.delta_prev = self.delta
        # adjust trust region
        if self.rho < 1/4:
            self.delta = self.delta / 4
            print(f'Reduce trust region: delta={self.delta}, rho={self.rho}')
        elif 0.4 <= self.rho <= 2 and 0.99 * self.delta <= np.linalg.norm(self.p):
            self.delta = min(2*self.delta, self.delta_max)
            print(f'Expand trust region: delta={self.delta}')
        else:
            print(f'Same trust region: delta={self.delta}')

        self.theta_prev = theta
        if self.rho <= self.eta:
            print(f'Not update theta')
        theta = theta + self.p

        self.iter_num += 1

        return theta
