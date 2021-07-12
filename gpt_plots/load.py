import csv
import glob

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

plt.rcParams.update({
    'axes.titlesize': 20,
    'axes.labelsize': 16,
    'font.family': 'serif',
    'font.sans-serif': ['Times'],
    'text.latex.preamble':
        r'\usepackage[T2A]{fontenc}'
        r'\usepackage[utf8]{inputenc}'
})

path = '/home/moby/PycharmProjects/datasets_processing/'
img_path = '/home/moby/PycharmProjects/datasets_processing/figures/'
models_raw = (
    'svm_melspec',
    'cnn_melspec_raw',
    'lstm_melspec_raw',
    # 'lstm_mfcc_raw',
)

models_aug = (
    'svm_melspec_aug',
    'cnn_melspec_aug',
    'lstm_melspec_aug',
    # 'lstm_mfcc_aug',
)

model_names = (
    'SVM',
    'CNN',
    'LSTM',
    # 'LSTM(MFCC)',
)

classes = (
    'Normal',
    'Muting',
    'Vibrato',
    'Pull-off',
    'Hammer-on',
    'Sliding',
    'Bending'
)


##1 -
def load_metrics(model_name):
    files = glob.glob(path + 'metrics/' + model_name + '/*')
    res = []
    for f in files:
        with open(f, 'r') as _f:
            for row in csv.reader(_f):
                res.append(row)
    return np.asarray(res, dtype=np.float64)


def load_matrices(model_name):
    files = glob.glob(path + 'cm/' + model_name + '/*')
    matrices = []
    for f in files:
        matrices.append(pd.read_csv(f, sep=',', header=None).values)
    return np.asarray(matrices)


def plot_cm(cm, names=classes, title=''):
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(len(names), len(names)))
    sns.set(font_scale=1.2)
    ax.tick_params(axis='both', rotation='default', which='major', labelsize=14)
    # ax.tick_params(axis='x', rotation=0, which='major', labelsize=13)
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=names, yticklabels=names, cbar=False, linewidths=1,
                linecolor='black',
                cmap=sns.cubehelix_palette(50, hue=0.05, rot=0, light=1, dark=0))
    plt.ylabel('Ожидаемые классы')
    plt.xlabel('Предсказанные классы')
    plt.tight_layout()
    plt.savefig(img_path + title + '.pdf')
    plt.show(block=False)


def plot_box_whiskers(data, names=model_names, ax1=None):
    data = list(map(lambda a: a * 100, data))
    bp = ax1.boxplot(data, vert=False, whis=10, patch_artist=True)
    plt.yticks(list(range(1, len(names) + 1)), names)
    plt.grid()
    plt.xlim(60, 95)
    ax1.set(xlabel="Общая точность, %")
    plt.tight_layout()
    return bp


def plot_accuracy_whiskers(models, name):
    accs = []
    for model in models:
        metrics = load_metrics(model)
        accs.append(metrics[:, 0])
    accs = np.asarray(accs)
    fig1, ax1 = plt.subplots()
    plot_box_whiskers(accs, ax1=ax1)
    plt.savefig(img_path + name + '.pdf')
    plt.show()


def set_bp_colors(bp, face='orange', alpha=1.0):
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], alpha=alpha)

    for patch in bp['boxes']:
        patch.set(alpha=alpha)
        patch.set_facecolor(face)


def plot_multi_whiskers():
    name = 'multi_whiskers'

    accs1 = []
    for model in models_raw:
        metrics = load_metrics(model)
        accs1.append(metrics[:, 0])
    accs1 = np.asarray(accs1)

    accs2 = []
    for model in models_aug:
        metrics = load_metrics(model)
        accs2.append(metrics[:, 0])
    accs2 = np.asarray(accs2)

    fig1, ax1 = plt.subplots()
    bp1 = plot_box_whiskers(accs1, ax1=ax1)
    set_bp_colors(bp1, face='grey', alpha=1.0)
    plt.savefig(img_path + name + '1.pdf')
    # plt.show()
    # ax2 = ax1.twinx()
    plt.grid()
    bp2 = plot_box_whiskers(accs2, ax1=ax1)
    set_bp_colors(bp1, face='grey', alpha=0.5)
    set_bp_colors(bp2, face='grey', alpha=1.0)
    plt.savefig(img_path + name + '2.pdf')
    plt.show()


# plot_multi_whiskers()


def plot_cms(models):
    for m in models:
        plot_cm(np.average(load_matrices(m), axis=0), title=m)


def metrics_avg(models):
    for model in models:
        print(model)
        metrics = load_metrics(model)
        print(metrics)
        print(np.average(metrics, axis=0) * 100)


def f1_avg(models):
    avgs = []
    for model in models:
        metrics = load_metrics(model)
        f1 = np.average(metrics, axis=0)[3]
        avgs.append(f1)
    return np.asarray(avgs)


def plot_f1_hist():
    raw = f1_avg(models_raw)
    aug = f1_avg(models_aug)
    x = np.arange(len(model_names))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, raw, width, label='Неизмененные данные')
    rects2 = ax.bar(x + width / 2, aug, width, label='Аугментированные данные')

    ax.set_ylabel('F-score')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()

    plt.ylim(0, 1.1)

    ax.bar_label(rects1, padding=3, fmt='%.2f')
    ax.bar_label(rects2, padding=3, fmt='%.2f')

    fig.tight_layout()
    plt.savefig(img_path + 'f_scores' + '.pdf')

    plt.show()


cm = np.average(load_matrices(models_raw[2]), axis=0)
plot_cm(cm, title='raw_cm')
