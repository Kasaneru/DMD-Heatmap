import numpy as np
import yaml
import sys
from scipy.optimize import curve_fit
import seaborn as sns
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

def fit_gaussian_2d(field2d, x, y):
    field2d = np.abs(field2d)
    field2d /= field2d.max()
    XX_, YY_ = np.meshgrid(x, y)
    xy_flat  = np.vstack([XX_.ravel(), YY_.ravel()])
    z_flat   = field2d.ravel()

    def gauss(xy, cx, cy, sigma, A):
        return A * np.exp(-((xy[0]-cx)**2 + (xy[1]-cy)**2) / (2*sigma**2))

    # Seed from peak
    peak_idx = np.unravel_index(field2d.argmax(), field2d.shape)
    p0 = [x[peak_idx[1]], y[peak_idx[0]], 0.08, 1.0]
    try:
        popt, _ = curve_fit(gauss, xy_flat, z_flat, p0=p0,
                            bounds=([0,0,1e-3,0],[1,1,0.5,1.5]),
                            maxfev=5000)
        return popt[0], popt[1], popt[2]   # cx, cy, sigma
    except Exception:
        return p0[0], p0[1], p0[2]

def load_config(file_path: str) -> dict:
    if not file_path:
        return {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Ошибка: файл '{file_path}' не найден.", file=sys.stderr)
    except yaml.YAMLError as e:
        print(f"Ошибка разбора YAML: {e}", file=sys.stderr)
    return {}

def read_sources_params(c: dict) -> list:
    s_params = []
    for _, p in c.get('sources', {}).items():
        s_params.append({'amp': p['amp'], 'x0': p['x0'], 'y0': p['y0'], 'width': p['width'], 'omega': p['omega'], 'phi': p['phi']})
    return s_params

def ids_of_n_smallest(arr, n, eps=0.1):
    indexed = [(val, idx) for idx, val in enumerate(arr)]
    indexed.sort(key=lambda x: x[0])
    
    groups = []  # каждый элемент: (значение_группы, список_индексов)
    current_group_vals = []
    current_group_indices = []
    
    for val, idx in indexed:
        if not current_group_vals:
            current_group_vals = [val]
            current_group_indices = [idx]
        else:
            if val - current_group_vals[-1] <= eps:
                current_group_vals.append(val)
                current_group_indices.append(idx)
            else:
                group_value = np.mean(current_group_vals)
                groups.append((group_value, current_group_indices))
                current_group_vals = [val]
                current_group_indices = [idx]
    
    if current_group_vals:
        group_value = np.mean(current_group_vals)
        groups.append((group_value, current_group_indices))
    
    groups.sort(key=lambda x: x[0])
    
    n = min(n, len(groups))
    result_indices = [min(indices) for _, indices in groups[:n]]
    return result_indices

def anim_heatmap(D, rec, n_x, n_y, to_save=''):
    def get_frame(i):
        axes[0, 0].cla()
        sns.heatmap(D[:, i].reshape(n_x, n_y),
                    ax = axes[0, 0],
                    cbar = True,
                    cbar_ax = axes[0,1],
                    vmin = D.min(),
                    vmax = D.max(),
                    cmap='hot')
        axes[1, 0].cla()
        sns.heatmap(rec[:, i].reshape(n_x, n_y),
                    ax = axes[1,0],
                    cbar = True,
                    cbar_ax = axes[1,1],
                    vmin = D.min(),
                    vmax = D.max(),
                    cmap='hot')
        axes[2, 0].cla()
        sns.heatmap(D[:, i].reshape(n_x, n_y) - rec[:, i].reshape(n_x, n_y),
                    ax = axes[2, 0],
                    cbar = True,
                    cbar_ax = axes[2,1],
                    vmin = D.min(),
                    vmax = D.max(),
                    cmap='hot')

    
    grid_kws = {'width_ratios': (0.9, 0.05), 'wspace': 0.2}
    fig, axes = plt.subplots(3, 2, gridspec_kw = grid_kws, figsize = (12, 12))
    anim = FuncAnimation(fig = fig, func = get_frame, frames = 100, interval = 1, blit = False)
    if to_save:
        print('Saving gif')
        anim.save(to_save+'.gif', writer='pillow', fps=30)
    else:
        plt.show()
