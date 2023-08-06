import os
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from scipy import integrate

from sklearn.decomposition import PCA

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets


def LSM(x, y):
    '''
    Solves the equation y = A @ x with least-squares minimization
    :param x: x
    :param y: f(x)
    :returns: coefficient matrix A
    '''
    return np.linalg.lstsq(x, y, rcond=10e-6)

def RBF(eps, domain):
    '''
    Defines a concatenation of L radial basis functions with L central
    points uniformly spaced over the given domain and denominator eps
    :param L: number of central points
    :param eps: denominator of the radial basis function formula
    :param domain: domain of the function to aproximate
    '''
    phi = lambda x: np.array([np.exp(-np.linalg.norm(x_l - x) ** 2 / eps) for x_l in domain])
    return phi
    
def solve_euler(f_ode, y0, time):
    """
    Solves the given ODE system in f_ode using forward Euler.
    :param f_ode: the right hand side of the ordinary differential equation d/dt x = f_ode(x(t)).
    :param y0: the initial condition to start the solution at.
    :param time: np.array of time values (equally spaced), where the solution must be obtained.
    :returns: (solution[time,values], time) tuple.
    """
    yt = np.zeros((len(time), len(y0)))
    yt[0, :] = y0
    step_size = time[1]-time[0]
    for k in range(1, len(time)):
        yt[k, :] = yt[k-1, :] + step_size * f_ode(yt[k-1, :])
    return yt, time

def phase_portrait(A, X, Y, title=""):
    """
    Plots a linear vector field in a streamplot, defined with X and Y coordinates and the matrix A.
    :param A: matrix that defines the vector field
    :param X: X coordinates of the streamplot
    :param Y: Y coordinates of the streamplot
    """
    UV = A@np.row_stack([X.ravel(), Y.ravel()])
    U = UV[0,:].reshape(X.shape)
    V = UV[1,:].reshape(X.shape)

    fig = plt.figure(figsize=(25, 25))
    gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 2])

    #  Varying density along a streamline
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.streamplot(X, Y, U, V, density=[1, 1])
    ax0.set_title(title)
    ax0.set_aspect(1)
    return ax0

def phase_portrait_nonlinear(C, phi, X, Y, title=""):
    """
    Plots a nonlinear vector field in a streamplot, defined with X and Y coordinates, the basis function phi and coefficient C
    :param C: coefficient matrix C used to calculate the vector v = C * phi(x)
    :param phi: radial basis function
    :param X: X coordinates of the streamplot
    :param Y: Y coordinates of the streamplot
    """
    grid_points = np.array([[X[i][j], Y[i][j]] for i in range(len(X)) for j in range(len(Y))])
    v_grid = np.array(C @ phi(grid_points[0]))
    for i in range(1, len(grid_points)):
        v_grid = np.vstack((v_grid, C @ phi(grid_points[i])))
    U, V = v_grid.T
    U = U.reshape(X.shape)
    V = V.reshape(X.shape)

    fig = plt.figure(figsize=(25, 25))
    gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 2])

    #  Varying density along a streamline
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.streamplot(X, Y, U, V, density=[1, 1])
    ax0.set_title(title)
    ax0.set_aspect(1)
    return ax0

def predict_x1(x0, t, A):
    """
    Solves the linear system ẋ = Ax for x0 as a initial point, up to time T_end = ∆t = t
    using the solve_ivp integration method.
    :param x0: initial point
    :param t: time
    :param A: estimated matrix that defines a linear vector field
    """
    def f(t, x):
        fx = A @ x
        return fx
    return integrate.solve_ivp(f, (0, t), x0, t_eval=np.array([t])).y

def predict_x1_nonlinear(x0, t, C, phi):
    """
    Solves the nonlinear system ẋ = Cphi(x) for x0 as a initial point, up to time T_end = ∆t = t
    using the solve_ivp integration method.
    :param x0: initial point
    :param t: time
    :param C: coefficient matrix C used to calculate the vector v = C * phi(x)
    :param phi: radial basis function
    """
    def f(t, x):
        fx = C @ phi(x)
        return fx
    return integrate.solve_ivp(f, (0, t), x0, t_eval=np.array([t])).y

def data_shift(data, delta_n):
    """
    Shift the given data series by delta_n and 2*delta_n of row numbers
    :param data: list
        data series to shift
    :param delta_n: int
        row numbers to shift
    :return [data0, data1, data2]: list
        shifted data columns
    :return index: list
        shifted index column
    """
    length = len(data)
    index = np.arange(length)
    if delta_n > 0:
        data0 = data[:-delta_n*2]
        data1 = data[delta_n:-delta_n]
        data2 = data[delta_n*2:]
        index = index[:-delta_n*2]
    else:
        delta_n = -delta_n
        data0 = data[delta_n*2:]
        data1 = data[delta_n:-delta_n]
        data2 = data[:-delta_n*2]
        index = index[delta_n*2:]
    return [data0, data1, data2], index       

def plot_shift(data, column_name, delta_n):
    """
    Calculate and plot the shifted data series.
    Based on Takens theorem, for a list of data x_0(n) we can use x_0(n+delta_n) and x_0(n+2*delta_n)
    together with it to capture the embedding of the original signal.
    :param data: list of 3 data series and a list of int as index
        shifted data series and index to plot
    :param column_name: str
        controlling names of the variables in the plot
    :param delta_n: int
        row numbers to shift
    :return: none
    """
    shifted_data, index = data
    length = len(index)
    # dynamically control the size of scatter points in the plot
    size = 1000 / np.log(length) **3

    fig = plt.figure(figsize=(20,6))

    ax0 = fig.add_subplot(131)
    ax0.scatter(shifted_data[0], shifted_data[1], c=index, cmap=plt.cm.twilight_shifted, s=size)
    ax0.set_xlabel('${}(n)$'.format(column_name))
    ax0.set_ylabel('${}(n+\Delta n)$'.format(column_name))
    ax0.set_title('${}(n)$ and ${}(n+\Delta n)$ in 2D space, $\Delta n=$'.format(column_name, column_name)
                 + str(delta_n))

    ax1 = fig.add_subplot(132)
    ax1.scatter(index, shifted_data[0], c=index, cmap=plt.cm.twilight_shifted, s=size)
    ax1.scatter(index, shifted_data[1], c=index, cmap=plt.cm.twilight_shifted, s=size/2)
    ax1.scatter(index, shifted_data[2], c=index, cmap=plt.cm.twilight_shifted, s=size/4)
    ax1.set_xlabel('$n$ (line number)')
    ax1.set_ylabel('${}(n)$, ${}(n+\Delta n)$, ${}(n+2\Delta n)$'.format(column_name, column_name, column_name))
    ax1.set_title('${}(n)$, ${}(n+\Delta n)$ and ${}(n+2\Delta n)$ against $n$, $\Delta n=$'.format(column_name, column_name, column_name)
                 + str(delta_n))

    ax2 = fig.add_subplot(133, projection='3d')
    ax2.scatter(shifted_data[0], shifted_data[1], shifted_data[2], c=index, cmap=plt.cm.twilight_shifted, s=size/2)
    ax2.set_xlabel('${}(n)$'.format(column_name))
    ax2.set_ylabel('${}(n+\Delta n)$'.format(column_name))
    ax2.set_zlabel('${}(n+2\Delta n)$'.format(column_name))
    ax2.set_title('${}(n)$, ${}(n+\Delta n)$ and ${}(n+2\Delta n)$ in 3D space, $\Delta n=$'.format(column_name, column_name, column_name)
                 + str(delta_n))

def lorenz(x, y, z, s=10, b=8/3, r=28):
    ''' 
    Definition of the derivative equation for the lorenz system.
    :param x, y, z: float
        current system value
    :param s, b, r: float
        lorenz model parameters
    :returns: float
        derivative result of x, y, z
    '''
    dx = s*(y - x)
    dy = r*x - y - x*z
    dz = x*y - b*z
    return dx, dy, dz

def lorenz_calculate(Tend, dt, x0, y0, z0, s, b, r):   
    '''
    Calculate and plot the 3D trajectory of the Lorenz model from the initial state x_0=(x0, y0, z0), 
    and parameters sigma=s, beta=b, rho=r, simulating within range of time from 0 to Tend.

    :param Tend: int
        Termination time of the simulation 
    :param dt: float
        time step of the simulation
    :param x0, y0, z0: float
        initial state value
    :param s, b, r: float
        lorenz model parameters
    :returns: x, y, z: np.array
        calculated coordinate series of the Lorenz system
    '''
    # Define the time step size and number of iterations
    num_iterations = int(Tend / dt)

    # Set up the arrays to store the values of x, y, and z
    x = np.empty(num_iterations + 1,)
    y = np.empty(num_iterations + 1,)
    z = np.empty(num_iterations + 1,)

    # Set the initial values of x, y, and z
    x[0], y[0], z[0] = (x0, y0, z0)
    # Iterate over the time steps and update the values of x, y, and z using the Lorenz system equations
    for i in range(num_iterations):
        dx, dy, dz = lorenz(x[i], y[i], z[i], s, b, r)
        if (dx>1e8 or dy>1e8 or dz>1e8):
            break
        x[i + 1] = x[i] + dx * dt
        y[i + 1] = y[i] + dy * dt
        z[i + 1] = z[i] + dz * dt
    
    return x, y, z