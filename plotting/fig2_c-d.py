# _*_ coding: utf-8 _*_
# Author: GC Zhu
# Email: zhugc2016@gmail.com

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from schizophrenia_detection import sz_dir


def power_func_duration(x, m, b):
    return m + b * x


def power_func_velocity(x, m, a):
    return m * (x ** a)


def sqrt_func_velocity(x, m):
    return m * np.sqrt(x)


def detect_outliers_gmm(data, n_components=2, threshold=0.05):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(data_scaled)
    densities = np.exp(gmm.score_samples(data_scaled))
    density_threshold = np.percentile(densities, threshold * 100)
    return densities >= density_threshold


def plot_duration(ax, df, color, title):
    df = df[df['duration'] < 150].dropna()
    data = df[['sac_amp', 'duration']].values

    if len(data) > 2:
        inliers_mask = detect_outliers_gmm(data)
        inliers = data[inliers_mask]

        ax.scatter(inliers[:, 0], inliers[:, 1], color=color, s=5, alpha=0.4)

        try:
            params, _ = curve_fit(power_func_duration, inliers[:, 0], inliers[:, 1])
            x_fit = np.linspace(0, 20, 100)
            ax.plot(x_fit, power_func_duration(x_fit, *params), 'k-', lw=0.5)
        except Exception as e:
            print(f"Duration fitting failed: {e}")

    ax.set_xlim(0, 15)
    ax.set_ylim(0, 150)
    ax.set_xlabel('Saccade Amplitude (°)')
    ax.set_ylabel('Saccade Duration (ms)')
    ax.set_title(title, fontsize=6)
    ax.grid(True, alpha=0.3)


def plot_velocity(ax, df, color, title):
    df = df[['sac_amp', 'peakv']].dropna()
    df = df[df['peakv'] < 1000]

    if len(df) > 2:
        ax.scatter(df['sac_amp'], df['peakv'], color=color, s=5, alpha=0.4)

        try:
            # params, _ = curve_fit(power_func_velocity, df['sac_amp'], df['peakv'])
            # x_fit = np.linspace(0, 20, 100)
            # ax.plot(x_fit, power_func_velocity(x_fit, *params), color='black', lw=0.5)
            params, _ = curve_fit(power_func_velocity, df['sac_amp'], df['peakv'])
            x_fit = np.linspace(0, 20, 100)
            ax.plot(x_fit, power_func_velocity(x_fit, *params), color='black', lw=0.5)
        except Exception as e:
            print(f"Velocity fitting failed: {e}")

    ax.set_xlim(0, 15)
    ax.set_ylim(0, 400)
    ax.set_xlabel('Saccade Amplitude (°)')
    ax.set_ylabel('Peak Velocity (°/s)')
    ax.set_title(title, fontsize=6)
    ax.grid(True, alpha=0.3)


def add_device_labels(fig, axes, labels):
    for ax, label in zip(axes[:, 0], labels):
        ax.annotate(label, xy=(-0.5, 0.5), xycoords='axes fraction',
                    ha='center', va='center', rotation=90,
                    fontsize=7, )


if __name__ == "__main__":
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'axes.linewidth': 0.5,
        'lines.linewidth': 0.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'axes.labelsize': 6,
        'figure.dpi': 1000,
        'figure.figsize': (3.54, 3)
    })

    output_dir = os.path.join(os.path.dirname(__file__), 'fig_2')
    os.makedirs(output_dir, exist_ok=True)

    subjects = [
        {'id_str': '132', 'sz': 0, 'label': 'HC', 'color': '#1f77b4', 'batch': 'batch_0'},
        {'id_str': '060', 'sz': 1, 'label': 'SZ', 'color': '#d62728', 'batch': 'batch_0'}
    ]

    devices = ['phone', 'eyelink', ]
    device_labels = ['Smartphone', 'EyeLink', ]
    task = 'fv'

    # ======= 图C：Peak Velocity vs Amplitude ========
    fig_c, axes_c = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False)
    for col, subj in enumerate(subjects):
        for row, device in enumerate(devices):
            subject_id = subj['id_str']
            label = subj['label']
            color = subj['color']
            try:
                data_file = os.path.join(
                    f'{sz_dir}/events', f'data_{device}', subj['batch'],
                    f"sac_{subject_id}_{task}.csv"
                )
                df = pd.read_csv(data_file) if os.path.exists(data_file) else pd.DataFrame()
                plot_velocity(axes_c[row, col], df, color, label)
            except Exception as e:
                print(f"Error in C plot: {subject_id} {device}: {e}")
    add_device_labels(fig_c, axes_c, device_labels)
    plt.tight_layout()
    fig_c.savefig(os.path.join(output_dir, 'fig2_c.tiff'), dpi=600, bbox_inches='tight')
    plt.close(fig_c)

    # ======= 图D：Duration vs Amplitude ========
    fig_d, axes_d = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False)
    for col, subj in enumerate(subjects):
        for row, device in enumerate(devices):
            subject_id = subj['id_str']
            label = subj['label']
            color = subj['color']
            try:
                data_file = os.path.join(
                    f'{sz_dir}/events', f'data_{device}', 'batch_0',
                    f"sac_{subject_id}_{task}.csv"
                )
                df = pd.read_csv(data_file) if os.path.exists(data_file) else pd.DataFrame()
                plot_duration(axes_d[row, col], df, color, label)
            except Exception as e:
                print(f"Error in D plot: {subject_id} {device}: {e}")
    add_device_labels(fig_d, axes_d, device_labels)
    plt.tight_layout()
    fig_d.savefig(os.path.join(output_dir, 'fig2_d.tiff'), dpi=600, bbox_inches='tight')
    plt.close(fig_d)
