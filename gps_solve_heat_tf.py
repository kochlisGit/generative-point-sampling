# --- Import Libraries ---

import os

# TODO: Remove if platform==Linux versions, because it works fine in linux setup. Windows/WSL only
os.environ['DDE_BACKEND'] = 'tensorflow.compat.v1'

import random
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from typing import Any, Callable, Dict, List, Optional, Tuple

# --- Configuration ---
pde_name = 'heat'

# PDE Config
x_min = -1.0
x_max = 1.0
t_min = 0.0
t_max = 1.0
a = 0.4

# PINN FCNet Config
seed = 0
num_inputs = 2
num_hiddens = 3
fc_units_per_hidden = 20
num_outputs = 1
activation = 'tanh'
weight_initializer = 'Glorot normal'
learning_rate = 1e-3
num_domain = 8000
num_boundaries = 1200
num_test = 10000
model_fcnet_directory = f'{pde_name}/checkpoints/fcnet'


def pde(x, y):
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t - a*dy_xx


def boundary_left_cond(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], x_min)


def boundary_left_value(x):
    return 0


def boundary_right_cond(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], x_max)


def boundary_right_value(x):
    return 0


def initial_cond(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[1], 0.0)


def initial_value(x):
    return np.sin(np.pi*x[:, 0:1])


# --- PINN Functions


def compute_errors(model: dde.Model, x_test: np.ndarray) -> np.ndarray:
    f = model.predict(x=x_test, operator=model.data.pde)
    return np.abs(f)


def init_pinn(net: dde.nn.FNN, data: dde.data.Data, save_directory: Optional[str] = None) -> dde.Model:
    if save_directory is None:
        save_filepath = None
    else:
        os.makedirs(save_directory, exist_ok=True)
        save_filepath = f'{save_directory}/net'

    model = dde.Model(data, net)
    model.compile('adam', lr=learning_rate)
    model.train(iterations=1, model_save_path=save_filepath)
    return model


def plot_error_points(points, errors, name):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot: x, y from points array, z is the errors, color also depends on errors
    scatter = ax.scatter(points[:, 0], points[:, 1], errors, c=errors, cmap='viridis', vmin=0, vmax=0.01)

    # Add a color bar which maps values to colors.
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Error magnitude')

    # Adding labels
    ax.set_title(name)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('Absolute Error')
    ax.set_zlim(0.0, 0.01)

    # Show plot
    plt.show()


def main():
    methods = [
        'baseline',
        'random-resampling',
        'rar',
        'genesis',
        'reps-greedy',
        'general'
    ]

    # --- Setting Seeds ---
    random.seed(seed)
    np.random.seed(seed=seed)
    tf.random.set_seed(seed=seed)

    # --- Generating Train Data ---
    geom = dde.geometry.Rectangle([x_min, t_min], [x_max, t_max])
    bc_left = dde.icbc.DirichletBC(geom, boundary_left_value, boundary_left_cond)
    bc_right = dde.icbc.DirichletBC(geom, boundary_right_value, boundary_right_cond)
    ic = dde.icbc.DirichletBC(geom, initial_value, initial_cond)
    data = dde.data.PDE(geom, pde, [bc_left, bc_right, ic], num_domain=num_domain, num_boundary=num_boundaries)

    x_test = geom.random_points(n=num_test)
    print(f'Generated {x_test.shape} points for {pde_name}')

    layers = [num_inputs] + [fc_units_per_hidden]*num_hiddens + [num_outputs]
    net = dde.nn.FNN(layers, activation, weight_initializer)

    # --- Initializing PINN ---
    model = init_pinn(net=net, data=data, save_directory=model_fcnet_directory)
    errors = compute_errors(model=model, x_test=x_test)
    err = np.mean(errors)
    print(f'# --- FCNet Initial MAE Evaluation on test set: {err:.6f} ---')

    # Restoring PINN ---

    for method in methods:
        last_iter = os.listdir(f'{pde_name}/checkpoints/{method}')[-1].split('-')[1].split('.')[0]
        restore_path = f'{pde_name}/checkpoints/{method}/net-{last_iter}.ckpt'

        print(f'Loading model from {restore_path}')

        model.restore(save_path=restore_path)

        # Compute errors
        errors = np.abs(model.predict(x_test, operator=pde))
        plot_error_points(points=x_test, errors=errors, name=method)


if __name__ == "__main__":
    main()
