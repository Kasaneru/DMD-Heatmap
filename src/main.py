import numpy as np
import matplotlib.pyplot as plt
from pydmd.hankeldmd import hankel_preprocessing
from dmd import DMD, hankel_preprocess, time_consistent_hankel_dmd
from pydmd import DMD as pyDMD
from pydmd.plotter import plot_summary
import warnings

warnings.filterwarnings('ignore')

def run_example(X, Y, svd_rank, t=None, dt=None, D=None, d=1):
    dmd = DMD(svd_rank=svd_rank)
    dmd.fit(X, Y)
    print('DMD eigenvalues:\n', dmd.eigs)
    if t is not None and dt is not None and D is not None:
        dmd.compute_dynamics(t, dt)
        dmd.plot_summary(t, dt, D, X, d=d)
    else:
        dmd.plot_singular(X)
        dmd.plot_eigs()
        dmd.plot_modes()

def check_example(D, svd_rank, t):
    '''compare to pyDMD'''
    dmd = pyDMD(svd_rank=svd_rank)
    delay_dmd = hankel_preprocessing(dmd, d=2)
    delay_dmd.fit(D)
    print('DMD eigenvalues:\n', dmd.eigs)
    plot_summary(delay_dmd, d=2)
    plt.plot(Y, label='original')
    plt.show()
    plt.plot(dmd.reconstructed_data[:,1:].real, label='reconstructed')
    plt.show()

# saddle focus equilibrium
def saddle_focus(t, y, rho = 1, omega = 1, gamma = 1):
    return np.dot(np.array([
                [-rho, -omega, 0],
                [omega, -rho, 0],
                [0, 0, gamma]
            ]), y)

def pendulum(t, y):
    th1, th2 = y
    dydt = [th2, -np.sin(th1)]
    return dydt


if __name__ == '__main__':
    # t = np.linspace(0, 2*np.pi, 100)
    # X = np.array([np.sin(t), np.cos(t), np.sin(2*t)]).T
    # Y = X[1:].T
    # X = X[:-1].T
    # run_example(X, Y, 3)

    x = np.linspace(-10, 10, 100)
    t = np.linspace(0, 6*np.pi, 80)
    dt = t[2] - t[1]
    Xm, Tm = np.meshgrid(x, t)
    f1 = np.multiply(20-0.2*np.pow(Xm, 2), np.exp((2.3j)*Tm))
    f2 = np.multiply(Xm, np.exp(0.6j*Tm))
    f3 = np.multiply(5*np.multiply(1/np.cosh(Xm/2), np.tanh(Xm/2)), 2*np.exp((0.1+2.8j)*Tm))
    D = (f1 + f2 + f3).T
    X = D[:,:-1]
    Y = D[:,1:]
    print('Example: 3 modes')
    run_example(X, Y, 3, t, dt, D)
    # # check_example(X, Y, 3, t)
    #
    # # Translational invariance
    # x = np.linspace(-10, 10, 50)
    # t = np.linspace(0, 10, 100)
    # dt = t[2] - t[1]
    # Xm,Tm = np.meshgrid(x, t)
    # D = np.exp(-np.pow((Xm-Tm+5)/2, 2))
    # D = D.T
    # X = D[:,:-1]
    # Y = D[:,1:]
    # print('Example: translational invariance rank=1')
    # run_example(X, Y, 1, t, dt, D)
    # print('Example: translational invariance rank=8')
    # run_example(X, Y, 8, t, dt, D)
    # # check_example(X, Y, 9, t)
    #
    # # Transient time behavior
    # x = np.linspace(-10, 10, 50)
    # t = np.linspace(0, 10, 100)
    # dt = t[2] - t[1]
    # Xm,Tm = np.meshgrid(x, t)
    # # create data with a single transient mode
    # D = np.exp(-np.pow((Xm)/4, 2)) * np.exp(-np.pow((Tm-5)/2, 2))
    # D = D.astype('complex').T
    # X = D[:,:-1]
    # Y = D[:,1:]
    # print('Example: transient time behavior')
    # run_example(X, Y, 1, t, dt, D)
    #
    # X = np.random.rand(3, 5000) * 10 - 5
    # Y = saddle_focus(0, X)
    # print('Example: saddle point (bad modes)')
    # run_example(X, Y, 3)

    # X = np.random.rand(2, 5000) * 10 - 5
    # Y = pendulum(0, X)
    # run_example(X, Y, 2)
    # check_example(X, Y, 2, t)

    def f4(x, t):
        return 1.0 / np.cosh(x + 3) * np.cos(2.3 * t)

    def f5(x, t):
        return 2.0 / np.cosh(x) * np.tanh(x) * np.sin(2.8 * t)

    nx = 65
    nt = 129
    x = np.linspace(-5, 5, nx)
    t = np.linspace(0, 4 * np.pi, nt)
    Xm, Tm = np.meshgrid(x, t)
    dt = t[1] - t[0]
    X1 = f4(Xm, Tm)
    X2 = f5(Xm, Tm)
    D = (X1 + X2).T
    X = D[:,:-1]
    Y = D[:,1:]
    d = 2
    # Xh, Yh = hankel_preprocess(X, Y, 2)
    Xh, Yh = time_consistent_hankel_dmd(X, Y, d=d)
    print('Example: needs hankel time-delay')
    # run_example(Xh, Yh, 4, t, dt, D, d=d)
    check_example(D, 4, t)
