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

def scale_mode(mode):
    min_val = np.min(mode)
    max_val = np.max(mode)
    if max_val == min_val:
        return np.zeros_like(mode, dtype=float)
    return 2.0 * (mode - min_val) / (max_val - min_val) - 1.0

def MSE(x, y):
    return np.mean((x - y)**2)

def heat_dmd(config_name: str, to_show=True):
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

    d = c['hankel']
    if d < 2:
        dmd.fit(D, t)
    else:
        delay_dmd = hankel_preprocessing(dmd, d=d)
        delay_t = t[: -d + 1]
        delay_dmd.fit(D, t=delay_t)

        dmd = delay_dmd

    save_dict = {
        'eigs': dmd.eigs,
        'modes': dmd.modes,
        'amplitudes': dmd.amplitudes,
        'dynamics': dmd.dynamics,
    }
    save_path = os.path.join('data/dmd/', config_name+'.npz')
    np.savez(save_path, **save_dict)

    rec = dmd.reconstructed_data.real
    print('Reconstruction Error')
    print('MSE: ', MSE(D, rec))

    if to_show:
        num_show = svd_rank
        fig, axs = plt.subplots(num_show,2, gridspec_kw={'hspace': 0.3})
        axs[0, 0].set_title('Real')
        axs[0, 1].set_title('Imag')
        for i in range(num_show):
            sns.heatmap(dmd.modes.real.T[i, :c['Ny']*c['Nx']].reshape(c['Nx'], c['Ny']),
                        ax = axs[i, 0],
                        xticklabels=False,
                        yticklabels=False)
                        # vmin=-0.1,
                        # vmax=0.1)
            sns.heatmap(dmd.modes.imag.T[i, :c['Ny']*c['Nx']].reshape(c['Nx'], c['Ny']),
                        ax = axs[i, 1],
                        xticklabels=False,
                        yticklabels=False)
                        # vmin=-0.1,
                        # vmax=0.1)
        plt.tight_layout()
        fig.show()

        fig, axs = plt.subplots(num_show,2, gridspec_kw={'wspace': 0.5, 'hspace': 0.3})
        axs[0, 0].set_title('Real')
        axs[0, 1].set_title('Imag')
        for i in range(num_show):
            axs[i, 0].plot(dmd.dynamics[i].real)
            axs[i, 0].set_xlabel('t')
            axs[i, 0].set_ylabel('T', rotation=0)
            axs[i, 1].plot(dmd.dynamics[i].imag)
            axs[i, 1].set_xlabel('t')
            axs[i, 1].set_ylabel('T', rotation=0)
        # plt.tight_layout()
        fig.show()
        plt.show()

if __name__ == '__main__':
    conf_name = sys.argv[1]
    if conf_name:
        heat_dmd(conf_name)
    else:
        print('Введите имя файла конфигурации без расширения!')
