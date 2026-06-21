import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import yaml
import os
import sys
from utils import load_config

def solve(conf_name: str, method='explicit', to_show=False, save=True, save_dir='data/heat_data/'):
    c_path = os.path.join('data/configs/', conf_name+'.yaml')
    c = load_config(c_path)

    x = np.linspace(0, c['Lx'], c['Nx'])
    y = np.linspace(0, c['Ly'], c['Ny'])
    dx = c['Lx'] / (c['Nx'] - 1)
    dy = c['Ly'] / (c['Ny'] - 1)
    a = c['a']
    T = c['T']
    dt = c['dt']
    Nt = int(T / dt)     # Количество шагов по времени

    X, Y = np.meshgrid(x, y)
    u = np.zeros((c['Nx'], c['Ny']))
    heat_sources = []
    ji = 0
    for name, p in c.get('sources', {}).items():
        # if ji == 0:
            # heat_sources.append(lambda t, p=p:
            #     (p['amp'] - t*10) * (1 + np.sin(p['omega'] * t + p['phi'])) / 2 * np.exp(-((X - p['x0'])**2 + (Y - p['y0'])**2) / (2 * p['width']**2))
            # )
        # else:
            heat_sources.append(lambda t, p=p:
                p['amp'] * (1 + np.sin(p['omega'] * t + p['phi'])) / 2 * np.exp(-((X - p['x0'])**2 + (Y - p['y0'])**2) / (2 * p['width']**2))
            )
        # ji += 1


    print(len(c.get('sources', {}).items()))

    def get_sources(t):
        return sum([heat_sources[i](t) for i in range(len(heat_sources))])
    save_interval = c['save_interval']
    saved_steps = Nt // save_interval
    u_data = np.zeros((saved_steps, c['Ny'], c['Nx']))

    cx = a * dt / dx**2
    cy = a * dt / dy**2
    if cx + cy > 0.5:
        print("Предупреждение: схема может быть неустойчивой!")

    for n in range(Nt):
        t = n * dt

        # Граничные условия
        u[0, :] = c['u_left']
        u[-1, :] = c['u_right']
        u[:, 0] = c['u_top']
        u[:, -1] = c['u_bottom']

        # Вычисление источников
        sources = get_sources(t)

        # Явная конечно-разностная схема
        u_new = u + cx * (np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) - 2*u) + \
                cy * (np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 2*u) + dt * sources # - dt * 0.1 * u

        u = u_new

        u += np.random.normal(c['mean'], c['variance'], size=(c['Nx'], c['Ny']))

        u[0, :] = c['u_left']
        u[-1, :] = c['u_right']
        u[:, 0] = c['u_top']
        u[:, -1] = c['u_bottom']

        # Сохранение данных
        if n % save_interval == 0:
            u_data[n // save_interval] = u

    if to_show:
        fig, axs = plt.subplots(1,5, gridspec_kw={'wspace': 0.3})
        for i in range(5):
            # cont = axs[i].contourf(u_data[20+i*20], levels=50, cmap='hot', origin='upper')
            axs[i].set_xlabel('x')
            axs[i].set_ylabel('y')
            sns.heatmap(u_data[20+i*20],
                        ax = axs[i],
                        xticklabels=False,
                        yticklabels=False,
                        vmin=0,
                        vmax=180,
                        cmap='hot')
        # plt.colorbar(cont, ax=axs[4])
        plt.tight_layout()
        plt.show()

    if save:
        name = c['Name'] + '.npy'
        save_path = os.path.join(save_dir, name)
        np.save(save_path, u_data)
        print(f"Данные сохранены в файл '{name}'")

if __name__ == '__main__':
    conf_name = sys.argv[1]
    if conf_name:
        solve(conf_name, to_show=True)
    else:
        print(f'Конфиг не найден!')
