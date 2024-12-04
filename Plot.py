import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d
from scipy.integrate import simps as simpson

from Env.save_tools import SaveIteration, SaveUAV, SaveUGV
from Utils.plot_utils import set_font_type_size, colorFader, lims


def rpo_iteration():
    iter_history = SaveIteration()
    iter_history.load('TR_iteration_True')
    iter_num = len(iter_history.theta_p)

    cm = plt.get_cmap('tab10')
    set_font_type_size()

    print(f'Initial theta:  ({iter_history.theta_p[0, 0]:.2f}, {iter_history.theta_p[0, 1]:.2f})\n'
          f'Terminal theta_p: ({iter_history.theta_p[-1, 0]:.2f}, {iter_history.theta_p[-1, 1]:.2f})\n'
          f'Terminal theta_s: ({iter_history.theta_s[-1, 0]:.2f}, {iter_history.theta_s[-1, 1]:.2f})')

    # Trajectory iteration
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    for i in range(iter_num):
        r_x_p = iter_history.traj_p.item().x[i]
        r_y_p = iter_history.traj_p.item().y[i]
        h_p = iter_history.traj_p.item().h[i]
        r_x_s = iter_history.traj_s.item().x[i]
        r_y_s = iter_history.traj_s.item().y[i]
        c_1 = colorFader('gray', cm.colors[0], i / (iter_num - 1))
        c_2 = colorFader('gray', cm.colors[3], i / (iter_num - 1))
        ax.plot3D(r_y_p, r_x_p, h_p, color=c_1, ls='--', linewidth=1.5)
        ax.plot3D(r_y_s, r_x_s, [0 * i for i in r_x_s], color=c_2, ls='-.', linewidth=1.5)
    ax.plot3D(r_y_p, r_x_p, h_p, color=c_1, ls='--', linewidth=1.5, label='UAV')
    ax.plot3D(r_y_s, r_x_s, [0 * i for i in r_x_s], color=c_2, ls='-.', linewidth=1.5, label='UGV')
    ax.grid(linestyle='--')
    # 设置图注样式
    ax.legend(loc=(0.7, 0.7), fancybox=True, edgecolor='black', framealpha=1)
    # 设置坐标注释
    ax.zaxis.set_rotate_label(False)
    ax.set_xlabel('Y-axis, m', labelpad=6)
    ax.set_ylabel('X-axis, m', labelpad=6)
    ax.set_zlabel('Altitude, m', rotation=90, labelpad=6)
    # 设置坐标轴
    ax.set_xlim(-50, 200)
    # ax.set_ylim(-50, 250)
    ax.set_zlim(0, 100)
    # 设置z轴位置
    ax.zaxis._axinfo['juggled'] = (1, 2, 0)
    # 设置比例
    limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
    ax.set_box_aspect(np.ptp(limits, axis=1))
    # 设置边框
    xlims, ylims, zlims = lims(ax.get_xlim()), lims(ax.get_ylim()), lims(ax.get_zlim())
    i = np.array([xlims[0], ylims[0], zlims[0]])
    f = np.array([xlims[0], ylims[0], zlims[1]])
    p = art3d.Poly3DCollection(np.array([[i, f]]))
    p.set_color('black')
    ax.add_collection3d(p)
    ax.xaxis.pane.set_edgecolor('#000000')
    ax.yaxis.pane.set_edgecolor('#000000')
    ax.zaxis.pane.set_edgecolor('#000000')
    ax.xaxis.pane.set_alpha(1)
    ax.yaxis.pane.set_alpha(1)
    ax.zaxis.pane.set_alpha(1)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # 设置角度
    ax.view_init(elev=33, azim=-47)
    # 设置尺寸
    # fig.set_size_inches(6.4, 6)
    # 调整边距
    fig.set_tight_layout(True)
    plt.savefig(f'Fig/Fig2_1a.tif', dpi=300)
    plt.savefig(f'Fig/Fig2_1a.pdf', dpi=300)

    # Rendezvous point iteration
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(range(iter_num), iter_history.theta_p[:, 0], color=cm.colors[0], ls='--', marker='o', linewidth=1.5, label='UAV', clip_on=False)
    ax1.plot(range(iter_num), iter_history.theta_s[:, 0], color=cm.colors[3], ls='-.', marker='o', linewidth=1.5, label='UGV', clip_on=False)
    ax2.plot(range(iter_num), iter_history.theta_p[:, 1], color=cm.colors[0], ls='--', marker='o', linewidth=1.5, clip_on=False)
    ax2.plot(range(iter_num), iter_history.theta_s[:, 1], color=cm.colors[3], ls='-.', marker='o', linewidth=1.5, clip_on=False)
    # 设置坐标注释
    plt.xlabel('Iteration', labelpad=6)
    ax1.set_ylabel(r'$\theta_x$, m', labelpad=6)
    ax2.set_ylabel(r'$\theta_y$, s', labelpad=6)
    # 设置图注样式
    ax1.legend(loc='best', fancybox=True, edgecolor='black', framealpha=1)
    # 设置坐标轴
    ax1.set_xlim(0, iter_num - 1)
    ax1.set_ylim(150, 250)
    ax1.set_yticks([150, 175, 200, 225, 250])
    ax1.set_xticks(range(iter_num))
    ax2.set_xlim(0, iter_num - 1)
    ax2.set_ylim(-20, 80)
    ax2.set_yticks([-20, 0, 20, 40, 60, 80])
    ax2.set_xticks(range(iter_num))
    # 显示
    plt.tight_layout()
    plt.savefig(f'Fig/Fig2_1b.tif', dpi=300)
    plt.savefig(f'Fig/Fig2_1b.pdf', dpi=300)

    # V and dV iteration
    fig, (ax1, ax2) = plt.subplots(2, 1)
    # axis-1
    ax1.plot(range(iter_num), iter_history.V_p, color=cm.colors[0], ls='--', marker='o', clip_on=False, label='UAV')
    ax1.plot(range(iter_num), iter_history.V_s, color=cm.colors[3], ls='-.', marker='o', clip_on=False, label='UGV')
    # axis-2
    dV_p_norm = [np.linalg.norm(iter_history.dV_p[i]) for i in range(iter_num)]
    dV_s_norm = [np.linalg.norm(iter_history.dV_s[i]) for i in range(iter_num)]
    ax2.plot(range(iter_num), dV_p_norm, color=cm.colors[0], ls='--', marker='o', clip_on=False)
    ax2.plot(range(iter_num), dV_s_norm, color=cm.colors[3], ls='-.', marker='o', clip_on=False)
    # 设置坐标注释
    plt.xlabel('Iteration', labelpad=6)
    ax1.set_yscale("log")
    ax1.set_ylabel('R', labelpad=6)
    ax2.set_yscale("log")
    ax2.set_ylabel(r'$\left\| \nabla R \right\|$', labelpad=6)
    # 设置图注样式
    ax1.legend(loc='best', fancybox=True, edgecolor='black', framealpha=1)
    # 设置坐标轴
    ax1.set_xlim(0, iter_num - 1)
    ax1.set_xticks(range(iter_num))
    ax1.set_ylim(1e1, 1e4)
    ax1.set_yticks([1e1, 1e2, 1e3, 1e4])
    ax2.set_xlim(0, iter_num - 1)
    ax2.set_xticks(range(iter_num))
    ax2.set_ylim(1e-2, 1e2)
    ax2.set_yticks([1e-2, 1e-1, 1e0, 1e1, 1e2])
    # 显示
    plt.tight_layout()
    plt.savefig(f'Fig/Fig2_2.tif', dpi=300)
    plt.savefig(f'Fig/Fig2_2.pdf', dpi=300)

    print(f'dR: {dV_p_norm[-1]}, {dV_s_norm[-1]}\n'
          f'R: {iter_history.V_p[-1]}')

    # p, delta iteration
    fig, (ax1, ax2) = plt.subplots(2, 1)
    # axis-1
    ds_p_norm = [np.linalg.norm(iter_history.ds_p[i]) for i in range(iter_num)]
    ds_s_norm = [np.linalg.norm(iter_history.ds_s[i]) for i in range(iter_num)]
    ax1.plot(range(iter_num), ds_p_norm, color=cm.colors[0], ls='--', marker='o', clip_on=False, label='UAV')
    ax1.plot(range(iter_num), ds_s_norm, color=cm.colors[3], ls='-.', marker='o', clip_on=False, label='UGV')
    # axis-2
    ax2.plot(range(iter_num), iter_history.delta_p, color=cm.colors[0], ls='--', marker='o', clip_on=False)
    ax2.plot(range(iter_num), iter_history.delta_s, color=cm.colors[3], ls='-.', marker='o', clip_on=False)
    # 设置坐标注释
    plt.xlabel('Iteration', labelpad=6)
    ax1.set_ylabel(r'$\|p\|$', labelpad=6)
    ax2.set_ylabel(r'$\Delta$', labelpad=6)
    # 设置图注样式
    ax1.legend(loc='best', fancybox=True, edgecolor='black', framealpha=1)
    # 设置坐标轴
    ax1.set_xlim(0, iter_num - 1)
    ax1.set_xticks(range(iter_num))
    ax1.set_ylim(0, 60)
    ax1.set_yticks([0, 20, 40, 60])
    ax2.set_xlim(0, iter_num - 1)
    ax2.set_xticks(range(iter_num))
    ax2.set_ylim(0, 100)
    ax2.set_yticks([0, 25, 50, 75, 100])
    # 显示
    plt.tight_layout()
    plt.savefig(f'Fig/Fig2_3.tif', dpi=300)
    plt.savefig(f'Fig/Fig2_3.pdf', dpi=300)


def wind_disturbance():
    cm = plt.get_cmap('tab10')
    set_font_type_size()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    data = np.load(f'Save/dryden_result.npz', allow_pickle=True)
    t_p = data['t_save']
    u_disturb = data['u_save']
    v_disturb = data['v_save']
    w_disturb = data['w_save']
    ax1.plot(t_p, u_disturb, color=cm.colors[0], linewidth=1)
    ax2.plot(t_p, v_disturb, color=cm.colors[3], linewidth=1)
    ax3.plot(t_p, w_disturb, color=cm.colors[2], linewidth=1)
    # 设置坐标注释
    plt.xlabel('Time, s', labelpad=6)
    ax1.set_ylabel('X-axis, m/s', labelpad=6)
    ax2.set_ylabel('Y-axis, m/s', labelpad=6)
    ax3.set_ylabel('Z-axis, m/s', labelpad=6)
    # 设置坐标轴
    ax1.set_xlim(0, 35)
    ax1.set_ylim(-0.2, 0.5)
    ax1.set_yticks([-0.2, 0, 0.2, 0.4])
    ax2.set_xlim(0, 35)
    ax2.set_ylim(-0.4, 0.2)
    ax2.set_yticks([-0.4, -0.2, 0, 0.2])
    ax3.set_xlim(0, 35)
    ax3.set_ylim(-0.5, 0.5)
    ax3.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
    # 设置尺寸
    fig.set_size_inches(6.4, 6)
    # 保存
    fig.set_tight_layout(True)
    plt.savefig(f'Fig/Fig2_4.tif', dpi=300)
    plt.savefig(f'Fig/Fig2_4.pdf', dpi=300)


def controlled_state():
    traj_1 = SaveUAV()
    traj_2 = SaveUGV()
    traj_1.load('sim_dec_traj_1')
    traj_2.load('sim_dec_traj_2')

    cm = plt.get_cmap('tab10')
    set_font_type_size()

    print(f't_f: {traj_1.t[-1]}')

def control_result_comparison():
    traj_1_dec = SaveUAV()
    traj_2_dec = SaveUGV()
    traj_1_cen = SaveUAV()
    traj_2_cen = SaveUGV()
    traj_1_dec.load('sim_dec_traj_1')
    traj_2_dec.load('sim_dec_traj_2')
    traj_1_cen.load('sim_cen_traj_1')
    traj_2_cen.load('sim_cen_traj_2')

    cm = plt.get_cmap('tab10')
    set_font_type_size()

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.plot3D(traj_1_dec.y, traj_1_dec.x, traj_1_dec.h, color=cm.colors[0], ls='-', linewidth=1.5, label='Proposed')
    ax.plot3D(traj_2_dec.y, traj_2_dec.x, [0 * i for i in traj_2_dec.x], color=cm.colors[0], ls='-', linewidth=1.5)
    ax.plot3D(traj_1_cen.y, traj_1_cen.x, traj_1_cen.h, color=cm.colors[3], ls='-', linewidth=1.5, label='Centralized')
    ax.plot3D(traj_2_cen.y, traj_2_cen.x, [0 * i for i in traj_2_cen.x], color=cm.colors[3], ls='-', linewidth=1.5)
    ax.grid(linestyle='--')
    # 设置图注样式
    ax.legend(loc=(0.7, 0.7), fancybox=True, edgecolor='black', framealpha=1)
    # 设置坐标注释
    ax.zaxis.set_rotate_label(False)
    ax.set_xlabel('Crossrange, m', labelpad=6)
    ax.set_ylabel('Downrange, m', labelpad=6)
    ax.set_zlabel('Altitude, m', rotation=90, labelpad=6)
    # 设置坐标轴
    ax.set_xlim(-150, 150)
    ax.set_zlim(0, 100)
    ax.set_zticks([0, 50, 100])
    # 设置z轴位置
    ax.zaxis._axinfo['juggled'] = (1, 2, 0)
    # 设置比例
    limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
    ax.set_box_aspect(np.ptp(limits, axis=1))
    # 设置边框
    xlims, ylims, zlims = lims(ax.get_xlim()), lims(ax.get_ylim()), lims(ax.get_zlim())
    i = np.array([xlims[0], ylims[0], zlims[0]])
    f = np.array([xlims[0], ylims[0], zlims[1]])
    p = art3d.Poly3DCollection(np.array([[i, f]]))
    p.set_color('black')
    ax.add_collection3d(p)
    ax.xaxis.pane.set_edgecolor('#000000')
    ax.yaxis.pane.set_edgecolor('#000000')
    ax.zaxis.pane.set_edgecolor('#000000')
    ax.xaxis.pane.set_alpha(1)
    ax.yaxis.pane.set_alpha(1)
    ax.zaxis.pane.set_alpha(1)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # 设置角度
    ax.view_init(elev=33, azim=-47)
    # 设置尺寸
    # fig.set_size_inches(6.4, 6)
    # 调整边距
    fig.set_tight_layout(True)
    plt.savefig(f'Fig/Fig2_5.tif', dpi=300)
    plt.savefig(f'Fig/Fig2_5.pdf', dpi=300)

    iter_history = SaveIteration()
    iter_history.load('TR_iteration_True')
    data = np.load(f'Save/computation_time_cen.npz', allow_pickle=True)
    time_g_list_cen = data['time_g_list']

    # compute sum u
    t_p = traj_1_dec.t
    T = traj_1_dec.T
    phi = traj_1_dec.phi
    CL = traj_1_dec.CL
    u_p_sum = [(i / 2) ** 2 + (j / np.radians(20)) ** 2 + (k / 0.7) ** 2 for i, j, k in zip(T, phi, CL)]
    t_s = traj_2_dec.t
    a_c = traj_2_dec.a_c
    eps_c = traj_2_dec.eps_c
    u_s_sum = [(i / 10) ** 2 + (j / np.radians(10)) ** 2 for i, j in zip(a_c, eps_c)]
    cost_dec = [simpson(u_p_sum, t_p), simpson(u_s_sum, t_s)]

    t_p = traj_1_cen.t
    T = traj_1_cen.T
    phi = traj_1_cen.phi
    CL = traj_1_cen.CL
    u_p_sum = [(i / 2) ** 2 + (j / np.radians(20)) ** 2 + (k / 0.7) ** 2 for i, j, k in zip(T, phi, CL)]
    t_s = traj_2_cen.t
    a_c = traj_2_cen.a_c
    eps_c = traj_2_cen.eps_c
    u_s_sum = [(i / 10) ** 2 + (j / np.radians(10)) ** 2 for i, j in zip(a_c, eps_c)]
    cost_cen = [simpson(u_p_sum, t_p), simpson(u_s_sum, t_s)]

    print(f'------Proposed------\n'
          f'Terminal theta: ({traj_1_dec.x[-1]:.2f}, {traj_1_dec.y[-1]:.2f})\n'
          f'Control energy: {cost_dec[0]}, {cost_dec[1]}\n'
          f'Terminal distance: {np.sqrt((traj_1_dec.x[-1] - traj_2_dec.x[-1])**2 + (traj_1_dec.y[-1] - traj_2_dec.y[-1])**2)}\n'
          f'Terminal angle: {np.abs(np.degrees(traj_1_dec.eps[-1] - traj_2_dec.eps[-1]))}')
    print(f'------Centralized------\n'
          f'Terminal theta: ({traj_1_cen.x[-1]:.2f}, {traj_1_cen.y[-1]:.2f})\n'
          f'Control energy: {cost_cen[0]}, {cost_cen[1]}\n'
          f'Terminal distance: {np.sqrt((traj_1_cen.x[-1] - traj_2_cen.x[-1]) ** 2 + (traj_1_cen.y[-1] - traj_2_cen.y[-1]) ** 2)}\n'
          f'Terminal angle: {np.abs(np.degrees(traj_1_cen.eps[-1] - traj_2_cen.eps[-1]))}\n')


def guidance_iteration():

    data = np.load(f'Save/opt_profile.npz', allow_pickle=True)
    theta_list = data['theta_list']
    J_list = data['J_list']
    iter_num = len(theta_list[0, :])

    cm = plt.get_cmap('tab10')
    set_font_type_size()

    fig, ax = plt.subplots()
    ax.plot(range(iter_num), J_list, color='k', ls='-.')
    for i in range(iter_num):
        c = colorFader(cm.colors[3], cm.colors[0], i / (iter_num - 1))
        ax.plot(i, J_list[i], color=c, marker='o', clip_on=False)
    # 设置坐标注释
    ax.set_xlabel('Iteration', labelpad=6)
    ax.set_ylabel(r'V', labelpad=6)
    # ax.annotate(r'$t_f$', (iter_num+1, J_list[-1] + 20))
    # 设置坐标轴
    ax.set_xlim(0, iter_num-1)
    ax.set_ylim(bottom=0)
    ax.set_xticks(range(1, iter_num))
    ax.set_yticks([0, 5, 10, 15, 20])
    # 保存
    plt.tight_layout()
    plt.savefig(f'Fig/Fig2_6.tif', dpi=300)
    plt.savefig(f'Fig/Fig2_6.pdf', dpi=300)


if __name__ == '__main__':
    rpo_iteration()
    wind_disturbance()
    controlled_state()
    control_result_comparison()
    guidance_iteration()
