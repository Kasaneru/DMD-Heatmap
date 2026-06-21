from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pydmd.hankeldmd import hankel_preprocessing
from pydmd import DMD, BOPDMD
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

def MAPE(x, y):
    return np.mean(np.abs((x - y) / x)) * 100

def dmd_diagnose(config_name: str):
    c_path = os.path.join('data/configs/', config_name+'.yaml')
    c = load_config(c_path)
    p = read_sources_params(c)
    svd_rank = c['modes']

    dmd_path = os.path.join('data/dmd/', config_name+'.npz')
    data = np.load(dmd_path)
    eigs = data['eigs']
    modes = data['modes']
    dynamics = data['dynamics']
    amplitudes = data['amplitudes']

    print('DMD eigenvalues:\n', eigs.shape)

    print(amplitudes)
    sources_num = len(c.get('sources', {}).items())
    sources_idx = ids_of_n_smallest(amplitudes, sources_num)
    print(sources_idx)
    # sources_idx = [0]

    # loc req
    x = np.linspace(0, c['Lx'], c['Nx'])
    y = np.linspace(0, c['Ly'], c['Ny'])
    locations = []
    for i, s in enumerate(sources_idx):
        mode_2d = modes.real.T[s, :c['Ny']*c['Nx']].reshape(c['Ny'], c['Nx'])
        cx_rec, cy_rec, sig_rec = fit_gaussian_2d(mode_2d, x, y)
        # print('AAA:', cx_rec, cy_rec, sig_rec)
        locations.append(np.array([cx_rec, cy_rec]))
    print('Localisation Error')
    for i in range(len(locations)):
        print(f'Loc_{i+1} MSE: {MSE(np.array([p[i]['x0'], p[i]['y0']]), locations[i])}')

    # Freq rec
    fs = 1 / (c['dt'] * c['save_interval'])

    freqs = []
    for i, s in enumerate(sources_idx):
        dyn = dynamics.real[s]
        f_est, amp_est = extract_frequency(dyn, fs)
        freqs.append(f_est)

    print('Frequency Error')
    for i, freq in enumerate(freqs):
        print(f'Freq_{i+1} MAPE: {MAPE(p[i]['omega'] / (2 * np.pi), freq)}')

    # amp req
    amps = []
    for i, s in enumerate(sources_idx):
        dyn = dynamics.real[s]
        amps.append(max(dyn) * 2 * (p[i]['omega'] / 4))
    # print(f'Amps: {amps}')

    print('Amplitude Error')
    for i in range(len(amps)):
        print(f'A_{i+1} MAPE: {MAPE(p[i]['amp'], amps[i])}')

    # for i in range(len(modes.T)):
    #     modes.real.T[i] = scale_mode(modes.real.T[i])
    #     modes.imag.T[i] = scale_mode(modes.imag.T[i])
    r_min, i_min, r_max, i_max = -1, -1, 1, 1

    locs = []
    for loc in locations:
        lx = int(loc[0] * 10)
        ly = int(loc[1] * 10)
        locs.append([lx, ly])
    print(f'Locs: {locs}')
    # locs = [[50, 50]]
    print(eigs)
    print(amplitudes)

    print('Phase Error')
    for i, s in enumerate(sources_idx):
        print('val:', modes.real.T[s, :c['Ny']*c['Nx']].reshape(c['Nx'], c['Ny'])[locs[i][0], locs[i][1]])
        phi_rec = np.pi * 5 * (modes.real.T[s, :c['Ny']*c['Nx']].reshape(c['Nx'], c['Ny'])[locs[i][0], locs[i][1]] + 0.1)
        print(phi_rec)
        print(f'Phi_{i+1} MSE: {MSE(p[i]['phi'], phi_rec)}')


if __name__ == '__main__':
    conf_name = sys.argv[1]
    if conf_name:
        dmd_diagnose(conf_name)
    else:
        print('Введите имя файла конфигурации без расширения!')

