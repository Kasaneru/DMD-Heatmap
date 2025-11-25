import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

class DMD:
    def __init__(
        self,
        svd_rank: int,
        exact=True
    ) -> None:
        self.svd_rank = svd_rank
        self.exact = exact
        self.eigs = None
        self.modes = None
        self.amplitudes = None
        self.dynamics = None
        self.data = None
        self.reconstructed_data = None

    def fit(self, X, Y):
        U2, Sig2, Vh2 = np.linalg.svd(X,False)
        r = self.svd_rank
        U = U2[:,:r]
        Sig = np.diag(Sig2)[:r,:r]
        V = Vh2.conj().T[:,:r]

        Atil = U.conj().T @ Y @ V @ np.linalg.inv(Sig)
        self.eigs, W = np.linalg.eig(Atil)

        if self.exact:
            Phi = Y @ V @ np.linalg.inv(Sig) @ W
        else:
            Phi = U @ W
        self.modes = Phi

        b = np.linalg.pinv(Phi) @ X[:,0]
        self.amplitudes = b

        return self.eigs, self.modes, self.amplitudes

    def compute_dynamics(self, t, dt):
        if self.eigs is None or self.amplitudes is None:
            raise TypeError('DMD is not initialized')
        Psi = np.zeros([self.svd_rank, len(t)], dtype='complex')
        for i, _t in enumerate(t):
            Psi[:,i] = np.multiply(np.pow(self.eigs, _t/dt), self.amplitudes)
        self.dynamics = Psi

    def check_result(self, data):
        if self.modes is None or self.dynamics is None:
            raise TypeError('DMD is not initialized')
        self.reconstructed_data = np.dot(self.modes, self.dynamics)
        # print('Close: ', np.allclose(data, self.reconstructed_data))
        fig, axs = plt.subplots(1, 2)
        axs[0].plot(data)
        axs[0].set_title('original')
        axs[1].plot(self.reconstructed_data)
        axs[1].set_title('reconstructed')
        plt.show()

    def plot_singular(self, X):
        U, Sig, Vh = np.linalg.svd(X,False)
        plt.title('Singular Values')
        plt.scatter([*range(len(Sig))], Sig)
        plt.show()

    def plot_eigs(self):
        if self.eigs is None:
            raise TypeError('DMD is not initialized')
        fig, ax = plt.subplots(1)
        ax.set_title('Eigenvalues')
        ax.axhline(0, color='black')
        ax.axvline(0, color='black')
        circ = Circle((0, 0), radius=1, fill=False, linestyle='--')
        ax.add_patch(circ)
        ax.scatter(self.eigs.real, self.eigs.imag, c='red', s=100)
        plt.show()

    def plot_modes(self):
        if self.modes is None:
            raise TypeError('DMD is not initialized')
        fig, ax = plt.subplots(1)
        ax.set_title('Modes')
        plt.plot(self.modes)
        plt.show()

    def plot_evolution(self, t, dt):
        if self.dynamics is None:
            raise TypeError('Dynamics are not computed')
        fig, axs = plt.subplots(len(self.dynamics), 1)
        plt.title('Dynamics')
        if len(self.dynamics) == 1:
            for i, mode_dynamics in enumerate(self.dynamics):
                plt.plot(mode_dynamics.real, label=f'Mode {i+1} Real')
                plt.plot(mode_dynamics.imag, label=f'Mode {i+1} Imag')
                plt.xlabel('Time')
                plt.legend()
                plt.grid(True)
        else:
            for i, mode_dynamics in enumerate(self.dynamics):
                axs[i].plot(mode_dynamics.real, label=f'Mode {i+1} Real')
                axs[i].plot(mode_dynamics.imag, label=f'Mode {i+1} Imag')
                axs[i].set_xlabel('Time')
                axs[i].legend()
                axs[i].grid(True)
        plt.show()

    def plot_summary(self, t, dt, data, X, d):
        if self.modes is None or self.dynamics is None:
            raise TypeError('DMD is not initialized')

        fig, axs = plt.subplots(3, max(4, len(self.modes.T)))

        U, Sig, Vh = np.linalg.svd(X,False)
        axs[0, 0].set_title('Singular Values')
        axs[0, 0].scatter([*range(len(Sig))], Sig)

        axs[0, 1].set_title('Eigenvalues')
        axs[0, 1].axhline(0, color='black')
        axs[0, 1].axvline(0, color='black')
        circ = Circle((0, 0), radius=1, fill=False, linestyle='--')
        axs[0, 1].add_patch(circ)
        axs[0, 1].scatter(self.eigs.real, self.eigs.imag, c='red', s=100)

        self.reconstructed_data = np.dot(self.modes, self.dynamics)
        # print('Close: ', np.allclose(data, self.reconstructed_data))
        axs[0, 2].plot(data)
        axs[0, 2].set_title('original')
        axs[0, 3].plot(self.reconstructed_data)
        axs[0, 3].set_title('reconstructed')

        for i, mode in enumerate(self.modes.T):
            axs[1, i].set_title(f'Mode {i+1}')
            axs[1, i].plot(mode)

        for i, mode_dynamics in enumerate(self.dynamics):
            axs[2, i].plot(mode_dynamics.real, label=f'Mode {i+1} Real')
            axs[2, i].plot(mode_dynamics.imag, label=f'Mode {i+1} Imag')
            axs[2, i].set_xlabel('Time')
            axs[2, i].legend()
            axs[2, i].grid(True)

        plt.show()

def hankel_preprocess(X, Y, m, matrix_type='row'):
    d, tau = X.shape

    if Y.shape != (d, tau):
        raise ValueError("Матрицы X и Y должны иметь одинаковую размерность")
    if m >= tau:
        raise ValueError("Глубина m должна быть меньше числа временных срезов tau")

    n = tau - m + 1

    H_X_blocks = []
    H_Y_blocks = []

    for i in range(d):
        window_X = np.lib.stride_tricks.sliding_window_view(X[i], m)
        H_X_i = window_X[:n].T  # Транспонируем чтобы получить (m, n)
        H_X_blocks.append(H_X_i)

        window_Y = np.lib.stride_tricks.sliding_window_view(Y[i], m)
        H_Y_i = window_Y[:n].T  # Транспонируем чтобы получить (m, n)
        H_Y_blocks.append(H_Y_i)

    if matrix_type == 'row':
        X_new = np.vstack(H_X_blocks)
        Y_new = np.vstack(H_Y_blocks)
    elif matrix_type == 'column':
        X_new = np.hstack(H_X_blocks)
        Y_new = np.hstack(H_Y_blocks)
    else:
        raise ValueError("matrix_type должен быть 'row' или 'column'")

    return X_new, Y_new

def time_consistent_hankel_dmd(X, Y, d):
    """
    Hankel-DMD с сохранением временной структуры
    """
    d, tau = X.shape
    
    # Вместо стандартного Hankel, используем подход с перекрывающимися окнами
    X_windows = []
    Y_windows = []
    
    for i in range(tau - d):
        X_window = X[:, i:i+d]  # (d, m)
        Y_window = Y[:, i:i+d]  # (d, m)
        
        # Реорганизуем в вектор (d*m,)
        X_windows.append(X_window.flatten('F'))  # column-major для сохранения структуры
        Y_windows.append(Y_window.flatten('F'))
    
    X_new = np.array(X_windows).T  # (d*m, tau-m)
    Y_new = np.array(Y_windows).T  # (d*m, tau-m)
    
    print(f"Time-consistent Hankel: {X_new.shape} -> {Y_new.shape}")
    return X_new, Y_new
