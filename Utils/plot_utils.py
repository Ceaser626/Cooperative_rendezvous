import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def set_font_type_size():
    medium_size = 14
    bigger_size = 16
    plt.rc('font', size=medium_size)  # controls default text sizes
    plt.rc('axes', titlesize=bigger_size)  # font size of the axes title
    plt.rc('axes', labelsize=bigger_size)  # font size of the x and y labels
    plt.rc('xtick', labelsize=bigger_size)  # font size of the tick labels
    plt.rc('ytick', labelsize=bigger_size)  # font size of the tick labels
    plt.rc('legend', fontsize=medium_size)  # legend font size
    plt.rc('font', family='Times New Roman')


def colorFader(c1, c2, mix=0):
    # fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1 = np.array(mpl.colors.to_rgb(c1))
    c2 = np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)


def lims(mplotlims):
    scale = 1.021
    offset = (mplotlims[1] - mplotlims[0]) * scale
    return mplotlims[1] - offset, mplotlims[0] + offset
