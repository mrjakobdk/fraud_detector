import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os import listdir

import json

from utils.flags import FLAGS


def find_csv_filenames(path_to_dir, suffix=".csv"):
    filenames = listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]


def plot_maker(plot_name):

    path = f'../plots/{plot_name}/'

    with open(path + "args") as json_file:
        args = json.load(json_file)
        colors = args["colors"]
        ylim = (float(args["ylim_min"]) - 0.01, float(args["ylim_max"]) + 0.01)
        y_label = args["y_label"]
        x_label = args["x_label"]

    X_SMALL_SIZE = 12
    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 18

    plt.rc('font', size=X_SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=X_SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    color_dict = {
        "blue": "#0077BB",
        "red": "#CC3311",
        "green": "#009988",
        "orange": "#FF7043",
        "pink": "#FF7DFF",
        "lightblue": "#33BBEE",
        "lightred": "#FF9696",
        "lightgreen": "#69D3C7",
        "grey": "#BBBBBB",
        "darkgrey": "#2F3129",
        "purple": "#C566E8"
    }

    if len(colors) == 0:
        colors = color_dict.keys()
    else:
        colors = colors.split()

    plt.ylim(ylim)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # plt.title("Simple Plot")

    plt.grid(True)

    ax = plt.subplot(111)

    x_max = 0

    for i, file in enumerate(find_csv_filenames(path)):
        print(file)
        df = pd.read_csv(path + file)
        x = df['Step'].tolist()
        y = df['Value'].tolist()
        color = color_dict[colors[i % len(colors)]]
        ax.plot(x, y, color, linewidth=2,
                label=file[:-4].replace("_", " "))  # , label='linear')
        x_max = max(x_max, np.max(x))

        if "val" in file and y_label == "accuracy":
            y_max_i = np.argmax(y)
            x_pos = x[y_max_i]
            plt.axvline(x=x_pos, linestyle="dashed", color=color)
            plt.text(x_pos, ylim[1], "best/val", rotation=0, horizontalalignment='center', verticalalignment='bottom', color=color)

    locs, _ = plt.xticks()
    while locs[-2] < x_max:
        locs, _ = plt.xticks()
        print(locs)
        dif = locs[-1] - locs[-2]
        x_lim = np.ceil(x_max / dif) * dif + 1
        print(x_lim)
        plt.xlim((0, x_lim))
        locs, _ = plt.xticks()

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.tick_params(color='#BDBDBD')

    plt.legend()

    plt.savefig("../plots/" + plot_name + ".png")  # , bbox_inches="tight")
    plt.show()


for name in FLAGS.plot_name.split():
    plot_maker(name)
