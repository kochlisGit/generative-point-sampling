# --- Import Libraries ---

import os

# TODO: Remove if platform==Linux versions, because it works fine in linux setup. Windows/WSL only
os.environ['DDE_BACKEND'] = 'tensorflow.compat.v1'

import pickle
import random
import time
import deepxde as dde
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from typing import Any, Callable, Dict, List, Optional, Tuple
from deap import base, creator, tools, algorithms
from deepxde.backend import tf as T
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import NoopLogger
from tqdm import tqdm

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
batch_size = None

# Training Config
num_domain = 8000
num_boundaries = 1200
trials = 100
iterations_per_trial = 1000
num_test = 10000

# Resampling Config
resampling_iterations = 100

# Multi-Sampling Config
num_sampling_points = 4

# GENESIS Population Config
population_size = 25
mutations_per_trial = 5

# Epsilon-Greedy Config
epsilon = 0.8

# GENERAL Config
general_batch_mode = 'complete_episodes'
general_episode_steps = 32
general_train_epochs = 10
general_critic = True
general_gae = True
general_lambda = 0.95
general_gamma = 0.99
general_sgd_minibatch_size = general_episode_steps
general_train_batch_size = general_episode_steps
general_shuffle_sequences = True
general_clip_param = 0.2
general_vf_loss_coeff = 0.5
general_learning_rate = 0.005

# --- Checkpoint Directories ---

model_fcnet_directory = f'{pde_name}/checkpoints/fcnet'
model_baseline_directory = f'{pde_name}/checkpoints/baseline'
model_random_resampling_directory = f'{pde_name}/checkpoints/random-resampling'
model_rar_directory = f'{pde_name}/checkpoints/rar'
model_ms_rar_directory = f'{pde_name}/checkpoints/ms-rar'
model_genesis_directory = f'{pde_name}/checkpoints/genesis'
model_reps_greedy_directory = f'{pde_name}/checkpoints/reps-greedy'
model_general_directory = f'{pde_name}/checkpoints/general'

# --- Experiments Directories ---

data_directory = f'{pde_name}/data'
experiments_baseline_directory = f'{pde_name}/experiments/baseline'
experiments_random_resampling_directory = f'{pde_name}/experiments/random-resampling'
experiments_rar_directory = f'{pde_name}/experiments/rar'
experiments_ms_rar_directory = f'{pde_name}/experiments/ms-rar'
experiments_genesis_directory = f'{pde_name}/experiments/genesis'
experiments_reps_greedy_directory = f'{pde_name}/experiments/reps-greedy'
experiments_general_directory = f'{pde_name}/experiments/general'

# --- PDE/BCs/ICs Functions ---


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


def train(
        model: dde.Model,
        learning_rate: float,
        batch_size: Optional[int],
        callbacks: Optional[list],
        trials: int,
        iterations_per_trial: int,
        x_test: np.ndarray,
        restore_path: Optional[str] = None,
        save_path: Optional[str] = None,
        sampling_fn: Optional[Callable] = None,
        sampling_options: Optional[Dict[str, Any]] = None,
        disregard_previous_anchors: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    error_history = []
    sampled_points = []
    execution_times = []
    best_score = np.inf
    best_trial = -1

    model.restore(save_path=restore_path)

    with tqdm(total=trials) as pbar:
        for trial in range(trials):
            # Evaluate PINN
            errors = compute_errors(model=model, x_test=x_test)
            err = np.mean(errors)

            if err < best_score:
                best_score = err
                best_trial = trial

            error_history.append(err)
            pbar.set_description(f'! --- Trial: {trial + 1}, MAE = {err:.6f} - (Best Score = {best_score:.6f} at Trial {best_trial}) ---')

            # Remove previous anchors (sampled points)
            if disregard_previous_anchors:
                model.data.anchors = None

            start = time.time()

            # Sample new points
            if sampling_fn is not None:
                new_points = sampling_fn(model=model, x_test=x_test, sampling_options=sampling_options)
                model.data.add_anchors(new_points)

                if new_points.shape[0] > 0:
                    if new_points.ndim == 1:
                        sampled_points.append(new_points)
                    else:
                        sampled_points.extend(new_points)

            # Train ADAM + L-BFGS
            model.compile(optimizer='adam', loss='MSE', lr=learning_rate)
            model.train(
                iterations=iterations_per_trial,
                batch_size=batch_size,
                callbacks=callbacks,
                disregard_previous_best=True
            )
            model.compile(optimizer='L-BFGS')
            model.train()

            end = time.time()
            pbar.update(1)
            execution_times.append(end-start)

    model.save(save_path=save_path)

    # Computing final evaluation
    error_history.append(np.mean(compute_errors(model=model, x_test=x_test)))
    execution_times.append(0.0)

    assert len(error_history) == len(execution_times)

    return np.float32(error_history), np.float32(sampled_points), np.float32(execution_times)


# --- RAR (Residual-based Adaptive Refinement Sampling) ---


def rar(model: dde.Model, x_test: np.ndarray, sampling_options: Dict[str, Any]) -> np.ndarray:
    n = sampling_options['num_sampling_points']
    errors = compute_errors(model=model, x_test=x_test)

    if n == 1:
        x_id = np.argmax(errors)
        return x_test[x_id]
    else:
        max_ids = np.argsort(errors.squeeze(axis=-1))[-n:]
        return x_test[max_ids]


# --- GENESIS (Genetic Sampling) ---


def evaluate_individual(individual, model:dde.Model):
    x_val = individual[0]
    t_val = individual[1]
    p = [[x_val, t_val]]

    error_value = -1*(np.abs(model.predict(p, operator=model.data.pde)))[0][0]
    return error_value,


def build_toolbox(model: dde.Model):
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMax)

    tlbox = base.Toolbox()
    tlbox.register('attr_x', random.uniform, x_min, x_max)
    tlbox.register('attr_t', random.uniform, t_min, t_max)
    tlbox.register('individual', tools.initCycle, creator.Individual, (tlbox.attr_x, tlbox.attr_t), n=1)
    tlbox.register('population', tools.initRepeat, list, tlbox.individual)
    tlbox.register("evaluate", evaluate_individual, model=model)
    tlbox.register('mate', tools.cxBlend, alpha=0.1)
    tlbox.register('mutate', tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    tlbox.register('select', tools.selTournament, tournsize=3)
    return tlbox


def mutate_population(population: List[Tuple[float, float]], toolbox, n_gen: int) -> List:
    for _ in range(n_gen):
        offspring = algorithms.varAnd(population=population, toolbox=toolbox, cxpb=0.5, mutpb=0.1)
        fits = toolbox.map(toolbox.evaluate, offspring)

        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))
    return population


def extract_best_individual(population: List[Tuple[float, float]]) -> np.ndarray:
    top_individuals = sorted(population, key=lambda ind: ind.fitness.values[0], reverse=True)[:1]
    additional_points = [(ind[0], ind[1]) for ind in top_individuals]
    return np.array([additional_points[0][0], additional_points[0][1]], dtype='float64')


def genesis(model: dde.Model, x_test: np.ndarray, sampling_options: Dict[str, Any]) -> np.ndarray:
    toolbox = sampling_options['toolbox']
    population_size = sampling_options['population_size']
    n_gen = sampling_options['mutations_per_trial']

    population = toolbox.population(n=population_size)
    population = mutate_population(population=population, toolbox=toolbox, n_gen=n_gen)
    return extract_best_individual(population=population)


# --- REPS-Greedy (Residual Sampling using Epsilon-Greedy) ---


def reps_greedy(model: dde.Model, x_test: np.ndarray, sampling_options: Dict[str, Any]) -> np.ndarray:
    eps = sampling_options['epsilon']

    # P(genesis) = epsilon, P(rar) = 1 - epsilon
    if random.random() > eps:
        return rar(model=model, x_test=x_test, sampling_options=sampling_options)
    else:
        point = genesis(
            model=model,
            x_test=x_test,
            sampling_options=sampling_options
        )
    return point


def train_agent(
        env_instance: type,
        x_test: np.ndarray,
        fcnet_directory: str,
        callbacks: Optional[list] = None
):
    ray.shutdown()
    ray.init(ignore_reinit_error=True)

    agent_config = PPOConfig()
    agent_config.model.update({
        'use_lstm': True,
        'vf_share_layers': True,
        'max_seq_len': general_episode_steps,
        'lstm_cell_size': 128
    })
    agent_config.framework(framework='tf')
    agent_config.rollouts(
        num_rollout_workers=1,
        batch_mode=general_batch_mode,
        rollout_fragment_length=general_episode_steps
    )
    agent_config.use_critic = general_critic
    agent_config.use_gae = general_gae
    agent_config.clip_param = general_clip_param
    agent_config.sgd_minibatch_size = general_sgd_minibatch_size
    agent_config.shuffle_sequences = general_shuffle_sequences
    agent_config.train_batch_size = general_episode_steps
    agent_config.vf_loss_coeff = general_vf_loss_coeff
    agent_config.seed = seed
    agent_config.gamma = general_gamma
    agent_config.lr = general_learning_rate
    agent_config.environment(disable_env_checking=True)
    agent_config.num_gpus = 0
    agent_config.logger_config = {'type': NoopLogger}

    data_config = {
        'x_min': x_min,
        'x_max': x_max,
        't_min': t_min,
        't_max': t_max,
        'a': a,
        'num_domain': num_domain,
        'num_boundary': num_boundaries,
        'x_test': x_test,
        'seed': seed
    }
    pinn_config = {
        'num_inputs': num_inputs,
        'hidden_layers': [fc_units_per_hidden]*num_hiddens,
        'num_outputs': num_outputs,
        'activation': activation,
        'weight_initializer': weight_initializer,
        'fcnet_directory': fcnet_directory,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'callbacks': callbacks,
        'adam_iterations': iterations_per_trial
    }
    general_config = {
        'episode_steps': general_episode_steps,
        'model_general_directory': model_general_directory,
        'experiments_general_directory': experiments_general_directory
    }

    env_config = {
        'data_config': data_config,
        'pinn_config': pinn_config,
        'general_config': general_config
    }
    agent = agent_config.environment(env=env_instance, env_config=env_config).build()

    for trial in range(trials):
        agent.train()


class GeneralEnv(gym.Env):
    def __init__(self, env_config: Optional[Dict] = None):
        super().__init__()

        self._x_min = env_config['data_config']['x_min']
        self._x_max = env_config['data_config']['x_max']
        self._t_min = env_config['data_config']['t_min']
        self._t_max = env_config['data_config']['t_max']
        self._a = env_config['data_config']['a']
        self._num_domain = env_config['data_config']['num_domain']
        self._num_boundary = env_config['data_config']['num_boundary']
        self._x_test = env_config['data_config']['x_test']
        self._seed = seed
        self._num_inputs = env_config['pinn_config']['num_inputs']
        self._hidden_layers = env_config['pinn_config']['hidden_layers']
        self._num_outputs = env_config['pinn_config']['num_outputs']
        self._activation = env_config['pinn_config']['activation']
        self._weight_initializer = env_config['pinn_config']['weight_initializer']
        self._fcnet_directory = env_config['pinn_config']['fcnet_directory']
        self._learning_rate = env_config['pinn_config']['learning_rate']
        self._batch_size = env_config['pinn_config']['batch_size']
        self._callbacks = env_config['pinn_config']['callbacks']
        self._adam_iterations = env_config['pinn_config']['adam_iterations']
        self._max_anchors = env_config['general_config']['episode_steps']
        self._model_general_directory = env_config['general_config']['model_general_directory']
        self._experiments_general_directory = env_config['general_config']['experiments_general_directory']

        random.seed(self._seed)
        np.random.seed(seed=self._seed)
        tf.random.set_seed(seed=self._seed)
        geom = self._geometry()
        icbc_list = self._boundaries(geom=geom)
        data = self._construct_data(geom=geom, icbc_list=icbc_list)
        self._model = self._construct_pinn(data=data)

        # PPO Output: One 2D Point
        self.action_space = gym.spaces.Box(
            low=np.float32([self._x_min, self._t_min]),
            high=np.float32([self._x_max, self._t_max]),
            dtype=np.float32,
            shape=(2,)
        )
        # PPO Input: Generated 2D Point, y, pde, step
        self.observation_space = gym.spaces.Box(
            low=np.float32([self._x_min, self._t_min, -np.inf, -np.inf]),
            high=np.float32([self._x_max, self._t_max, np.inf, np.inf]),
            dtype=np.float32,
            shape=(4,)
        )

        self._sampled_points = []
        self._sampled_points_per_trial = []
        self._eval_loss_per_trial = []
        self._step_counter = 0
        self._trial = 0

    def _geometry(self):
        return dde.geometry.Rectangle([self._x_min, self._t_min], [self._x_max, self._t_max])

    def _pde(self, x, y):
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        return dy_t - a*dy_xx

    @staticmethod
    def _boundaries(geom: dde.geometry.Geometry) -> List[dde.icbc.BC]:
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

        bc_left = dde.icbc.DirichletBC(geom, boundary_left_value, boundary_left_cond)
        bc_right = dde.icbc.DirichletBC(geom, boundary_right_value, boundary_right_cond)
        ic = dde.icbc.DirichletBC(geom, initial_value, initial_cond)
        return [bc_left, bc_right, ic]

    def _construct_data(self, geom: dde.geometry.Geometry, icbc_list: List[dde.icbc.BC]) -> dde.data.Data:
        return dde.data.PDE(geom, self._pde, icbc_list, num_domain=num_domain, num_boundary=num_boundaries)

    def _construct_pinn(self, data: dde.data.Data) -> dde.Model:
        layers = [self._num_inputs] + self._hidden_layers + [self._num_outputs]
        net = dde.nn.FNN(layers, self._activation, self._weight_initializer)
        model = dde.Model(data, net)
        model.compile(optimizer='adam', lr=self._learning_rate)
        model.train(iterations=1)
        model.restore(save_path=f'{self._fcnet_directory}/net-1.ckpt')
        return model

    def _eval_model(self) -> float:
        f = self._model.predict(x=self._x_test, operator=self._model.data.pde)
        return np.mean(np.abs(f))

    def _train_model(self):
        self._model.compile(optimizer='adam', lr=self._learning_rate)
        self._model.train(
            iterations=self._adam_iterations,
            batch_size=self._batch_size,
            callbacks=self._callbacks,
            disregard_previous_best=True
        )
        self._model.compile(optimizer='L-BFGS')
        self._model.train()
        self._eval_loss_per_trial.append(self._eval_model())

    def _write_history_log(self):
        pd.DataFrame({
            'Trials': range(len(self._eval_loss_per_trial)),
            'Mean Absolute Error': self._eval_loss_per_trial,
        }).to_csv(f'{self._experiments_general_directory}/history.csv', index=False)

        columns = [f'x{i//2 + 1}' if i % 2 == 0 else f't{(i+1)//2}' for i in range(len(self._sampled_points)*2)]
        points = np.float32(self._sampled_points_per_trial)
        pd.DataFrame(
            data=points.reshape(points.shape[0], -1),
            columns=columns
        ).to_csv(f'{self._experiments_general_directory}/anchors.csv', index=False)

    def _construct_observation(self, action: Optional[np.ndarray]=None):
        if action is None:
            x_rand = random.uniform(a=self._x_min, b=self._x_max)
            t_rand = random.uniform(a=self._t_min, b=self._t_max)
            action = np.float32([x_rand, t_rand])

        sampled_point = np.float32([action])
        u = self._model.predict(sampled_point, operator=None)[0, 0]
        sol = self._model.predict(x=sampled_point, operator=self._model.data.pde)[0, 0]
        return np.float32([
            action[0],
            action[1],
            u,
            sol
        ])

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> Tuple[np.ndarray, dict[str, Any]]:
        self._trial += 1
        self._step_counter = 0
        self._sampled_points = []
        self._model.data.anchors = None
        return self._construct_observation(action=None)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        self._step_counter += 1

        self._model.data.add_anchors(action)
        self._sampled_points.append(action)

        # Generating next input
        next_obs = self._construct_observation(action=action)

        # If batch is completed Then train, reward the agent and restart episode (sampling process). Finally store log
        # Else continue episode (sampling process) and no reward is provided
        if self._step_counter < self._max_anchors:
            reward = 0.0
            done = False
        else:
            self._train_model()
            eval_error = self._eval_model()

            self._eval_loss_per_trial.append(eval_error)
            self._sampled_points_per_trial.append(self._sampled_points)
            self._write_history_log()
            self._model.save(save_path=f'{self._model_general_directory}/net')

            print(f'Trial: {self._trial}, Iteration: {iterations_per_trial*self._trial}, Test MAE = {eval_error}')

            # !Change this if positive rewards do not work. In that case, the agent will try to reinforce PINN model.
            reward = float(eval_error)
            done = True

        # Return transition tuple: next observation, reward, done, info=None
        return next_obs, reward, done, {}

    def render(self):
        raise NotImplementedError('Render function is not supported')


def main():
    # --- Setting Seeds ---
    random.seed(seed)
    np.random.seed(seed=seed)
    tf.random.set_seed(seed=seed)

    # --- Generating Train Data ---
    geom = dde.geometry.Rectangle([x_min, t_min], [x_max, t_max])
    bc_left = dde.icbc.DirichletBC(geom, boundary_left_value, boundary_left_cond)
    bc_right = dde.icbc.DirichletBC(geom, boundary_right_value, boundary_right_cond)
    ic = dde.icbc.DirichletBC(geom, initial_value, initial_cond)
    data = dde.data.PDE(geom, pde, [bc_left, bc_right, ic], num_domain=2500, num_boundary=200)

    # --- Generating Test Points ---
    if not os.path.exists(f'{data_directory}/test.csv'):
        x_test = geom.random_points(n=num_test)
        test_df = pd.DataFrame({'x': x_test[:, 0], 't': x_test[:, 1]})
        os.makedirs(name=data_directory, exist_ok=True)
        test_df.to_csv(f'{data_directory}/test.csv', index=False)
        sns.scatterplot(data=test_df, x='x', y='t').set(title='Random Uniform Test Points')
        plt.savefig(f'{data_directory}/test.png')
        plt.clf()
    else:
        x_test = pd.read_csv(f'{data_directory}/test.csv').to_numpy()

        print(f'Found existing data file at {data_directory}/test.csv. Reusing {x_test.shape} test inputs')

    # --- Generating PINN Model ---
    layers = [num_inputs] + [fc_units_per_hidden]*num_hiddens + [num_outputs]
    net = dde.nn.FNN(layers, activation, weight_initializer)

    # --- Initializing PINN ---
    model = init_pinn(net=net, data=data, save_directory=model_fcnet_directory)
    errors = compute_errors(model=model, x_test=x_test)
    err = np.mean(errors)
    print(f'# --- FCNet Initial MAE Evaluation on test set: {err:.6f} ---')

    # --- Storing Train Data ---
    if os.path.exists(f'{data_directory}/train.csv'):
        first_point = pd.read_csv(f'{data_directory}/train.csv').to_numpy()[0]

        if not (first_point[0] == model.data.train_x_all[0, 0] and first_point[1] == model.data.train_x_all[0, 1]):
            print(
                f'Warning: Existing train data were found at {data_directory}/train.csv, '
                'but are different from current train data. Old data will be overwritten.'
            )

    pd.DataFrame(
        data=model.data.train_x_all,
        columns=['x', 't']
    ).to_csv(f'{data_directory}/train.csv', index=False)

    # # --- Baseline PINN ---
    #
    # os.makedirs(model_baseline_directory, exist_ok=True)
    # error_history, _, execution_times = train(
    #     model=model,
    #     learning_rate=learning_rate,
    #     batch_size=batch_size,
    #     callbacks=None,
    #     trials=trials,
    #     iterations_per_trial=iterations_per_trial,
    #     x_test=x_test,
    #     restore_path=f'{model_fcnet_directory}/net-{1}.ckpt',
    #     save_path=f'{model_baseline_directory}/net',
    #     sampling_fn=None,
    #     sampling_options=None,
    #     disregard_previous_anchors=False
    # )
    #
    # print(f'# --- Baseline MAE Evaluation on test set: {error_history[-1]:.6f} ---')
    #
    # os.makedirs(name=experiments_baseline_directory, exist_ok=True)
    # history_df = pd.DataFrame({
    #     'Trials': range(error_history.shape[0]),
    #     'Mean Absolute Error': error_history,
    #     'Time': execution_times
    # })
    # history_df.to_csv(f'{experiments_baseline_directory}/history.csv', index=False)
    # sns.lineplot(data=history_df, x='Trials', y='Mean Absolute Error').set(title='Baseline Evaluation')
    # plt.savefig(f'{experiments_baseline_directory}/history.png')
    # plt.clf()
    #
    # # --- Random Random-Resampling PINN ---
    #
    # os.makedirs(model_random_resampling_directory, exist_ok=True)
    # error_history, _, execution_times = train(
    #     model=model,
    #     learning_rate=learning_rate,
    #     batch_size=batch_size,
    #     callbacks=[dde.callbacks.PDEPointResampler(period=resampling_iterations)],
    #     trials=trials,
    #     iterations_per_trial=iterations_per_trial,
    #     x_test=x_test,
    #     restore_path=f'{model_fcnet_directory}/net-{1}.ckpt',
    #     save_path=f'{model_random_resampling_directory}/net',
    #     sampling_fn=None,
    #     sampling_options=None,
    #     disregard_previous_anchors=False
    # )
    #
    # print(f'# --- Random-Resampling MAE Evaluation on test set: {error_history[-1]:.6f} ---')
    #
    # os.makedirs(name=experiments_random_resampling_directory, exist_ok=True)
    # history_df = pd.DataFrame({
    #     'Trials': range(error_history.shape[0]),
    #     'Mean Absolute Error': error_history,
    #     'Time': execution_times
    # })
    # history_df.to_csv(f'{experiments_random_resampling_directory}/history.csv', index=False)
    # sns.lineplot(data=history_df, x='Trials', y='Mean Absolute Error').set(title='Random-Resampling Evaluation')
    # plt.savefig(f'{experiments_random_resampling_directory}/history.png')
    # plt.clf()
    #
    # # --- RAR (Residual Adaptive Refinement) ---
    #
    # os.makedirs(model_rar_directory, exist_ok=True)
    # error_history, sampled_points, execution_times = train(
    #     model=model,
    #     learning_rate=learning_rate,
    #     batch_size=batch_size,
    #     callbacks=None,
    #     trials=trials,
    #     iterations_per_trial=iterations_per_trial,
    #     x_test=x_test,
    #     restore_path=f'{model_fcnet_directory}/net-{1}.ckpt',
    #     save_path=f'{model_rar_directory}/net',
    #     sampling_fn=rar,
    #     sampling_options={'num_sampling_points': 1},
    #     disregard_previous_anchors=False
    # )
    #
    # print(f'# --- RAR Evaluation on test set: {error_history[-1]:.6f}, Num Sampled Points: {len(sampled_points)} ---')
    #
    # os.makedirs(name=experiments_rar_directory, exist_ok=True)
    # history_df = pd.DataFrame({
    #     'Trials': range(error_history.shape[0]),
    #     'Mean Absolute Error': error_history,
    #     'Time': execution_times
    # })
    # history_df.to_csv(f'{experiments_rar_directory}/history.csv', index=False)
    # sns.lineplot(data=history_df, x='Trials', y='Mean Absolute Error').set(title='RAR Evaluation')
    # plt.savefig(f'{experiments_rar_directory}/history.png')
    # plt.clf()
    #
    # sampled_points_df = pd.DataFrame({'Trials': range(error_history.shape[0] - 1), 'x': sampled_points[:, 0], 't': sampled_points[:, 1]})
    # sampled_points_df.to_csv(f'{experiments_rar_directory}/anchors.csv', index=False)
    # plt.xlim(x_min, x_max)
    # plt.ylim(t_min, t_max)
    # sns.scatterplot(data=sampled_points_df, x='x', y='t').set(title='RAR Sampled Points')
    # plt.savefig(f'{experiments_rar_directory}/anchors.png')
    # plt.clf()
    #
    # # --- Multi-Sampling RAR ---
    #
    # os.makedirs(model_ms_rar_directory, exist_ok=True)
    # error_history, sampled_points, execution_times = train(
    #     model=model,
    #     learning_rate=learning_rate,
    #     batch_size=batch_size,
    #     callbacks=None,
    #     trials=trials,
    #     iterations_per_trial=iterations_per_trial,
    #     x_test=x_test,
    #     restore_path=f'{model_fcnet_directory}/net-{1}.ckpt',
    #     save_path=f'{model_ms_rar_directory}/net',
    #     sampling_fn=rar,
    #     sampling_options={'num_sampling_points': num_sampling_points},
    #     disregard_previous_anchors=False
    # )
    #
    # print(f'# --- Multi-Sampling RAR Evaluation on test set: {error_history[-1]:.6f} ---')
    #
    # os.makedirs(name=experiments_ms_rar_directory, exist_ok=True)
    # history_df = pd.DataFrame({
    #     'Trials': range(error_history.shape[0]),
    #     'Mean Absolute Error': error_history,
    #     'Time': execution_times
    # })
    # history_df.to_csv(f'{experiments_ms_rar_directory}/history.csv', index=False)
    # sns.lineplot(data=history_df, x='Trials', y='Mean Absolute Error').set(title='Multi-Sampling RAR Evaluation')
    # plt.savefig(f'{experiments_ms_rar_directory}/history.png')
    # plt.clf()
    #
    # reshaped_points = np.reshape(a=sampled_points, newshape=(trials, num_sampling_points*2))
    # sampled_points_df = pd.DataFrame(
    #     data=reshaped_points,
    #     columns=[f'x{i//2 + 1}' if i % 2 == 0 else f't{(i+1)//2}' for i in range(num_sampling_points*2)]
    # )
    # sampled_points_df.insert(loc=0, column='Trials', value=pd.Series(range(trials)))
    # sampled_points_df.to_csv(f'{experiments_ms_rar_directory}/anchors.csv', index=False)
    # plt.xlim(x_min, x_max)
    # plt.ylim(t_min, t_max)
    # point_data = np.reshape(
    #     a=sampled_points_df.drop(columns=['Trials']).values,
    #     newshape=(trials*num_sampling_points, 2)
    # )
    # point_data_df = pd.DataFrame(data=point_data, columns=['x', 't'])
    # sns.scatterplot(data=point_data_df, x='x', y='t').set(title='Multi-Sampling RAR Sampled Points')
    # plt.savefig(f'{experiments_ms_rar_directory}/anchors.png')
    # plt.clf()
    #
    # # --- GENESIS (Genetic Sampling) ---
    #
    # # Initialize Toolbox, Population
    toolbox = build_toolbox(model=model)
    #
    # os.makedirs(model_genesis_directory, exist_ok=True)
    # sampling_options = {'toolbox': toolbox, 'population_size': population_size, 'mutations_per_trial': mutations_per_trial}
    # error_history, sampled_points, execution_times = train(
    #     model=model,
    #     learning_rate=learning_rate,
    #     batch_size=batch_size,
    #     callbacks=None,
    #     trials=trials,
    #     iterations_per_trial=iterations_per_trial,
    #     x_test=x_test,
    #     restore_path=f'{model_fcnet_directory}/net-{1}.ckpt',
    #     save_path=f'{model_genesis_directory}/net',
    #     sampling_fn=genesis,
    #     sampling_options=sampling_options,
    #     disregard_previous_anchors=False
    # )
    #
    # print(f'# --- GENESIS Evaluation on test set: {error_history[-1]:.6f} ---')
    #
    # os.makedirs(name=experiments_genesis_directory, exist_ok=True)
    # history_df = pd.DataFrame({
    #     'Trials': range(error_history.shape[0]),
    #     'Mean Absolute Error': error_history,
    #     'Time': execution_times
    # })
    # history_df.to_csv(f'{experiments_genesis_directory}/history.csv', index=False)
    # sns.lineplot(data=history_df, x='Trials', y='Mean Absolute Error').set(title='GENESIS RAR Evaluation')
    # plt.savefig(f'{experiments_genesis_directory}/history.png')
    # plt.clf()
    #
    # sampled_points_df = pd.DataFrame({'Trials': range(error_history.shape[0] - 1), 'x': sampled_points[:, 0], 't': sampled_points[:, 1]})
    # sampled_points_df.to_csv(f'{experiments_genesis_directory}/anchors.csv', index=False)
    # plt.xlim(x_min, x_max)
    # plt.ylim(t_min, t_max)
    # sns.scatterplot(data=sampled_points_df, x='x', y='t').set(title='GENESIS Sampled Points')
    # plt.savefig(f'{experiments_genesis_directory}/anchors.png')
    # plt.clf()
    #
    # --- REPS-Greedy (Residual Sampling using Epsilon-Greedy) ---

    os.makedirs(model_reps_greedy_directory, exist_ok=True)
    error_history, sampled_points, execution_times = train(
        model=model,
        learning_rate=learning_rate,
        batch_size=batch_size,
        callbacks=None,
        trials=trials,
        iterations_per_trial=iterations_per_trial,
        x_test=x_test,
        restore_path=f'{model_fcnet_directory}/net-{1}.ckpt',
        save_path=f'{model_reps_greedy_directory}/net',
        sampling_fn=reps_greedy,
        sampling_options={
            'toolbox': toolbox,
            'population_size': population_size,
            'mutations_per_trial': mutations_per_trial,
            'epsilon': epsilon,
            'num_sampling_points': 1
        },
        disregard_previous_anchors=False
    )

    print(f'# --- REPS-Greedy Evaluation on test set: {error_history[-1]:.6f} ---')

    os.makedirs(name=experiments_reps_greedy_directory, exist_ok=True)
    history_df = pd.DataFrame({
        'Trials': range(error_history.shape[0]),
        'Mean Absolute Error': error_history,
        'Time': execution_times
    })
    history_df.to_csv(f'{experiments_reps_greedy_directory}/history.csv', index=False)
    sns.lineplot(data=history_df, x='Trials', y='Mean Absolute Error').set(title='REPS-Greedy Evaluation')
    plt.savefig(f'{experiments_reps_greedy_directory}/history.png')
    plt.clf()

    sampled_points_df = pd.DataFrame({'Trials': range(error_history.shape[0] - 1), 'x': sampled_points[:, 0], 't': sampled_points[:, 1]})
    sampled_points_df.to_csv(f'{experiments_reps_greedy_directory}/anchors.csv', index=False)
    plt.xlim(x_min, x_max)
    plt.ylim(t_min, t_max)
    sns.scatterplot(data=sampled_points_df, x='x', y='t').set(title='REPS Sampled Points')
    plt.savefig(f'{experiments_reps_greedy_directory}/anchors.png')
    plt.clf()

    # # --- GENERAL ---
    #
    # os.makedirs(name=model_general_directory, exist_ok=True)
    # os.makedirs(name=experiments_general_directory, exist_ok=True)
    # train_agent(
    #     env_instance=GeneralEnv,
    #     x_test=pd.read_csv(f'{data_directory}/test.csv').to_numpy(),
    #     fcnet_directory=model_fcnet_directory,
    #     callbacks=None
    # )
    #
    # history_df = pd.read_csv(f'{experiments_general_directory}/history.csv')
    # sns.lineplot(data=history_df, x='Trials', y='Mean Absolute Error').set(title='REPS-Greedy Evaluation')
    # plt.savefig(f'{experiments_general_directory}/history.png')
    # plt.clf()
    #
    # eval_error = history_df.iloc[-1]['Mean Absolute Error']
    # print(f'# --- GENERAL Evaluation on test set: {eval_error:.6f} ---')
    #
    # sampled_points_per_trial = pd.read_csv(f'{experiments_general_directory}/anchors.csv').to_numpy()
    # initial_sampled_points = np.reshape(a=sampled_points_per_trial[0], newshape=(general_episode_steps, 2))
    # initial_df = pd.DataFrame({'x': initial_sampled_points[:, 0], 't': initial_sampled_points[:, 1]})
    # sns.scatterplot(data=initial_df, x='x', y='t').set(title='GENERAL Initial Sampled Points')
    # plt.savefig(f'{experiments_general_directory}/initial_anchors.png')
    # plt.clf()
    #
    # final_sampled_points = np.reshape(a=sampled_points_per_trial[-1], newshape=(general_episode_steps, 2))
    # final_df = pd.DataFrame({'x': final_sampled_points[:, 0], 't': final_sampled_points[:, 1]})
    # sns.scatterplot(data=final_df, x='x', y='t').set(title='GENERAL Final Sampled Points')
    # plt.savefig(f'{experiments_general_directory}/final_anchors.png')
    # plt.clf()


if __name__ == "__main__":
    main()
