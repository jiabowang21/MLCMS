import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sympy import Symbol, core
from sympy.solvers import solve
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets


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

def plot_phase_portrait(A, X, Y, title=""):
    """
    Plots a linear vector field in a streamplot, defined with X and Y coordinates and the matrix A.
    """
    UV = A@np.row_stack([X.ravel(), Y.ravel()])
    U = UV[0,:].reshape(X.shape)
    V = UV[1,:].reshape(X.shape)

    fig = plt.figure(figsize=(25, 25))
    gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 2])

    #  Varying density along a streamline
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.streamplot(X, Y, U, V, density=[0.5, 1])
    ax0.set_title(title)
    ax0.set_aspect(1)
    return ax0

def phase_portrait(w, A, y0, time, title=""):
    '''
    Calls the Euler's method and plots the phase portrait with the trajectory over it
    :param w: the width of the grid.
    :param A: the vector field.
    :param y0: the initial condition to start the solution at.
    :param time: np.array of time values (equally spaced), where the solution must be obtained.
    :param title: the title to be displayed above the plot
    :returns: (solution[time,values], time) tuple.
    '''
    Y, X = np.mgrid[-w:w:100j, -w:w:100j]
    # Euler's method to construct and plot a trajectory over the stream plot
    yt, time = solve_euler(lambda y: A@y, y0, time)

    # linear vector field A*x
    ax0 = plot_phase_portrait(A, X, Y, title)
    ax0.set_xlim(-w, w)
    ax0.set_ylim(-w, w)
    
    # then plot the trajectory over it
    ax0.plot(yt[:, 0], yt[:, 1], c='red')

    # prettify
    ax0.set_aspect(1)
    
def bifurcation_diagram(function: str, a_min, a_max, precision):
    '''
    Creates and plots the 1D bifurcation diagram of the function in the range [min_a, max_a] incrementing by a_step
    :param function: the evolution function of the dynamical system.
    :param a_min: the minimum value that can achieve alpha.
    :param a_max: the maximum value that can achieve alpha.
    :param precision: incrementing each alpha by this value.
    '''
    x = Symbol('x')
    alphas = np.round(np.arange(a_min, a_max + precision, precision), 2)
    solutions = {}
    for alpha in alphas:
        sol = solve(eval(function), x)
        if (abs(sol[0]) == 0):
            sol.append(core.numbers.Integer(0))
        for i, single_sol in enumerate(sol):
            if isinstance(single_sol, core.numbers.Float) or isinstance(single_sol, core.numbers.Integer):
                if i not in solutions:
                    solutions[i] = [single_sol]
                else:
                    solutions[i].append(single_sol)
            else:
                if i not in solutions:
                    solutions[i] = [None]
                else:
                    solutions[i].append(None)
    fig = plt.figure()
    ax = fig.add_subplot()
    colors = ['red', 'blue']
    for i in solutions.keys():
        ax.plot(alphas, solutions[i], c = colors[i])
        ax.legend(['Unstable', 'Stable'])
    plt.title("dynamical system described by $\dot{x}$ = " + function)
    plt.xlim(alphas[0], alphas[-1])
    ax.set_xlabel('alpha')
    ax.set_ylabel('x')
    plt.show()
    
def phase_portrait_system(system: list, alpha: float):
    """
    Plots the phase portrait of the given 'system', where 'system' is a 2 dimensional system given as couple of strings
    :param system: system ODEs
    :param alpha: system's parameter
    """
    w = 3
    Y, X = np.mgrid[-w:w:100j, -w:w:100j]
    U, V = [], []
    for x2 in X[0]:
        for x1 in Y[:, 0]:
            u = eval(system[0])
            v = eval(system[1])
            U.append(u)
            V.append(v)
    U = np.reshape(U, X.shape)
    V = np.reshape(V, X.shape)
    
    fig = plt.figure(figsize=(5, 5))

    ax0 = fig.add_subplot()
    ax0.streamplot(X, Y, U, V, density=2)
    ax0.set_aspect(1)
    ax0.set_title(r'$\alpha={0:.2f}$'.format(alpha))
    return ax0
    
def solve_euler_system(system, alpha, y0, time):
    """
    Solves the given ODE system using forward Euler.
    :param system: system ODEs
    :param y0: the initial condition to start the solution at.
    :param time: np.array of time values (equally spaced), where the solution must be obtained.
    :returns: (solution[time,values], time) tuple.
    """
    yt = np.zeros((len(time), len(y0)))
    yt[0, :] = y0
    step_size = time[1]-time[0]
    for k in range(1, len(time)):
        x1 = yt[k-1,0]
        x2 = yt[k-1,1]
        u = eval(system[0])
        v = eval(system[1])
        yt[k, :] = yt[k-1, :] + step_size * np.array([u, v])
    return yt, time

def cusp_bifurcation():
    """
    Creates 3D and 2D plots of the cusp bifurcation 
    """
    # Sample (x, a2) uniformly
    a2_samples = [round(a2, 2) for a2 in np.random.uniform(0, 1.5, 50000)]
    x_samples = [round(x, 2) for x in np.random.uniform(-1.5, 1.5, 50000)]
    a1_samples = []
    solutions = {}
    for x, a2 in zip(x_samples, a2_samples):
        # Calculates a1 for every (x, a2)
        a1 = round(-a2 * x + x ** 3, 2)
        a1_samples.append(a1)
        
        # Keeps records of x for every (a1, a2)
        key = (a1, a2)
        if key in solutions:
            solutions[key].add(x)
        else:
            solutions[key] = {x}
    
    def create_axes(angle1, angle2, position):
        """
        Adds a 3D subplot to a figure given the angles of visualization and the position
        This subplot scatters the different samples 'a1_samples', 'a2_samples' and 'x_samples'
        :param angle1: first angle of visualization
        :param angle2: second angle of visualization
        :param position: position of the subplot in the figure
        """
        ax = fig.add_subplot(1, 3, position, projection='3d')
        ax.scatter(a1_samples, a2_samples, x_samples, cmap = "viridis", c = a2_samples)
        ax.set_xlabel(r'$\alpha_1$')
        ax.set_ylabel(r'$\alpha_2$')
        ax.set_zlabel(r'$x$')
        ax.view_init(angle1, angle2)
        return ax
    
    # 3D plot
    fig = plt.figure(figsize=plt.figaspect(0.3))
    ax1 = create_axes(-90, 90, 1)
    ax2 = create_axes(120, -90, 2)
    ax3 = create_axes(-120, 90, 3)
    plt.show()
    
    # 2D plot
    fig = plt.figure()
    # Creates a colormap for every point (a1, a2)
    # If it has more than one solution x, a red point will be plotted, blue otherwise
    colors = ["red" if len(solutions[(a1, a2)]) > 1 else "blue" for a1, a2 in zip(a1_samples, a2_samples)]
    ax2d = plt.axes()
    ax2d.set_xlabel(r'$\alpha_1$')
    ax2d.set_ylabel(r'$\alpha_2$')
    ax2d.scatter(a1_samples, a2_samples, s=1, c=colors)
    plt.show()
    