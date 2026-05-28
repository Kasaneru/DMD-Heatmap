from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pydmd.hankeldmd import hankel_preprocessing
from pydmd import DMD, BOPDMD
from pydmd.plotter import plot_summary
from utils import load_config, ids_of_n_smallest, read_sources_params, fit_gaussian_2d
import sys
import os

import warnings
warnings.filterwarnings("ignore")

def extract_frequency(signal, sample_rate):
    n = len(signal)
    yf = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=1/sample_rate)
    amplitudes = np.abs(yf) / n

    idx_max = np.argmax(amplitudes[1:]) + 1
    freq = freqs[idx_max]
    amp = amplitudes[idx_max]
    return freq, amp

def scale_mode(mode):
    min_val = np.min(mode)
    max_val = np.max(mode)
    if max_val == min_val:
        return np.zeros_like(mode, dtype=float)
    return 2.0 * (mode - min_val) / (max_val - min_val) - 1.0

def MSE(x, y):
    return np.mean((x - y)**2)

def MAPE(x, y, eps=0.0001):
    return np.mean(np.abs((x - y) / (x+eps))) * 100

def dmd_diagnose(config_name: str, show=False):
    c_path = os.path.join('data/configs/', config_name+'.yaml')
    c = load_config(c_path)
    p = read_sources_params(c)
    print(p)

    n_t = int(c['T'] / c['dt'] // c['save_interval'])
    n_x = int(c['Nx'])
    n_y = int(c['Ny'])
    d_path = os.path.join('data/heat_data/', config_name+'.npy')
    D = np.load(d_path).reshape(n_t, n_x * n_y).T

    t = np.arange(n_t)

    svd_rank = c['modes']

    dmd = BOPDMD(svd_rank=svd_rank, num_trials=0)
    dmd.fit(D, t)

    # d = 2
    # delay_dmd = hankel_preprocessing(dmd, d=d)
    # delay_t = t[: -d + 1]
    # delay_dmd.fit(D, t=delay_t)
    #
    # dmd = delay_dmd

    print('DMD eigenvalues:\n', dmd.eigs.shape)

    # print(dmd.amplitudes)
    sources_num = len(c.get('sources', {}).items())
    sources_idx = ids_of_n_smallest(dmd.amplitudes, sources_num)
    print(sources_idx)

    # Sytem rec
    rec = dmd.reconstructed_data.real
    print('MSE: ', MSE(D, rec))

    # loc req
    x = np.linspace(0, c['Lx'], c['Nx'])
    y = np.linspace(0, c['Ly'], c['Ny'])
    locations = []
    for i, s in enumerate(sources_idx):
        mode_2d = dmd.modes.real.T[s].reshape(c['Ny'], c['Nx'])
        cx_rec, cy_rec, sig_rec = fit_gaussian_2d(mode_2d, x, y)
        # print('AAA:', cx_rec, cy_rec, sig_rec)
        locations.append(np.array([cx_rec, cy_rec]))
    for i in range(len(locations)):
        print(f'Loc_{i+1} MSE: {MSE(np.array([p[i]['x0'], p[i]['y0']]), locations[i])}')

    # Freq rec
    fs = 1 / (c['dt'] * c['save_interval'])

    freqs = []
    # fq = {}
    for i, s in enumerate(sources_idx):
        dyn = dmd.dynamics.real[s]
        f_est, amp_est = extract_frequency(dyn, fs)
        freqs.append(f_est)
        # fq[s] = f_est
    # print(freqs)
    for i, freq in enumerate(freqs):
        print(f'Freq_{i+1} MAPE: {MAPE(p[i]['omega'] / (2 * np.pi), freq)}')

    # amp req
    amps = []
    for i, s in enumerate(sources_idx):
        dyn = dmd.dynamics.real[s]
        amps.append(max(dyn) * 2 * (p[i]['omega'] / 4))
    # print(f'Amps: {amps}')
    for i in range(len(amps)):
        print(f'A_{i+1} MAPE: {MAPE(p[i]['amp'], amps[i])}')

    for i in range(len(dmd.modes.T)):
        dmd.modes.real.T[i] = scale_mode(dmd.modes.real.T[i])
        dmd.modes.imag.T[i] = scale_mode(dmd.modes.imag.T[i])
    r_min, i_min, r_max, i_max = -1, -1, 1, 1

    locs = []
    for loc in locations:
        lx = int(loc[0] * 10)
        ly = int(loc[1] * 10)
        locs.append([lx, ly])
    for i, s in enumerate(sources_idx):
        phi_rec = np.pi / 2 * (dmd.modes.real.T[s].reshape(c['Nx'], c['Ny'])[locs[i][0], locs[i][1]] + 1)
        print(f'Phi_{i+1} MAPE: {MAPE(p[i]['phi'], phi_rec)}')

    if show:
        num_show = svd_rank
        fig, axs = plt.subplots(num_show,2, gridspec_kw={'hspace': 0.3})
        axs[0, 0].set_title('Real')
        axs[0, 1].set_title('Imag')
        for i in range(num_show):
            sns.heatmap(dmd.modes.real.T[i].reshape(100, 100)[25:75],
                        ax = axs[i, 0],
                        xticklabels=False,
                        yticklabels=False,
                        vmin=r_min,
                        vmax=r_max)
            sns.heatmap(dmd.modes.imag.T[i].reshape(100, 100)[25:75],
                        ax = axs[i, 1],
                        xticklabels=False,
                        yticklabels=False,
                        vmin=i_min,
                        vmax=i_max)
        plt.tight_layout()
        fig.show()

        fig, axs = plt.subplots(num_show,2, gridspec_kw={'wspace': 0.5, 'hspace': 0.5})
        axs[0, 0].set_title('Real')
        axs[0, 1].set_title('Imag')
        for i in range(num_show):
            axs[i, 0].plot(dmd.dynamics[i].real)
            axs[i, 0].set_xlabel('t')
            axs[i, 0].set_ylabel('T', rotation=0)
            axs[i, 1].plot(dmd.dynamics[i].imag)
            axs[i, 1].set_xlabel('t')
            axs[i, 1].set_ylabel('T')
        # plt.tight_layout()
        fig.show()
        plt.show()

if __name__ == '__main__':
    conf_name = sys.argv[1]
    if conf_name:
        dmd_diagnose(conf_name, show=True)
    else:
        print('Введите имя файла конфигурации без расширения!')

