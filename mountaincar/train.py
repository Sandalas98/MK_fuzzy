import logging
import tempfile
import dill
from copy import deepcopy

import click
import mlflow

import gym
import gym_mountain_car

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lcs.agents import EnvironmentAdapter
from lcs.agents.acs2 import Configuration, ACS2
from lcs.metrics import population_metrics

logging.basicConfig(level=logging.INFO)


def print_cl(cl):
    actions = ['L', '-', 'R']
    action = actions[cl.action]

    marked = ''

    if cl.is_marked():
        marked = '(*)'

    return f"{cl.condition} - {action} - {cl.effect} [fit: {cl.fitness:.3f}, r: {cl.r:.2f}, q: {cl.q:.2f}, exp: {cl.exp}, num: {cl.num} {marked}]"


def log_population_artifact(filename, population):
    with tempfile.TemporaryDirectory() as tmpdir:

        # Readable population
        with open(f"{tmpdir}/{filename}.txt", mode="w") as f:
            # Dump data
            for cl in sorted(population, key=lambda c: -c.fitness):
                f.write(f"{print_cl(cl)}\n")

            f.flush()
            mlflow.log_artifact(f.name)

        # Serialized population
        with open(f"{tmpdir}/{filename}.dill", mode="wb") as f:
            dill.dump(population, f)
            f.flush()
            mlflow.log_artifact(f.name)


def log_metrics_artifact(filename, metrics):
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(f"{tmpdir}/{filename}.dill", mode="wb") as f:
            dill.dump(metrics, f)
            f.flush()
            mlflow.log_artifact(f.name)


def log_metrics(prefix, metrics):
    mlflow.log_metric(f"{prefix}-avg-steps-in-trial", metrics['steps_in_trial'].mean())
    mlflow.log_metric(f"{prefix}-avg-reward", metrics['reward'].mean())


def merge_metrics(m1, m2, cfg):
    tmp_metrics_1 = pd.DataFrame(m1)
    tmp_metrics_2 = pd.DataFrame(m2)

    tmp_metrics_1['type'] = 'explore'
    tmp_metrics_2['type'] = 'explore-decay'

    # get last trial value
    last_trial = tmp_metrics_1.tail(1)['trial'].tolist()[0]

    # update metrics from the second phase
    tmp_metrics_2['trial'] += last_trial + cfg.metrics_trial_frequency

    metrics_df = tmp_metrics_1.append(tmp_metrics_2)
    metrics_df.set_index('trial', inplace=True)
    return metrics_df


def metrics_to_df(m):
    metric_df = pd.DataFrame(m)
    metric_df.set_index('trial', inplace=True)
    return metric_df


def plot_steps_in_trial(file, metrics_df, window):
    metrics_df['steps_in_trial'].rolling(window=window).mean().plot(figsize=(14, 6),
                                                                    title='Averaged steps in each trial')
    with tempfile.TemporaryDirectory() as tmpdir:
        plt.savefig(f"{tmpdir}/{file}")
        mlflow.log_artifact(f"{tmpdir}/{file}")


def plot_avg_fitness(file, metrics_df, window):
    fig, ax = plt.subplots(figsize=(14, 6))
    metrics_df['avg_fitness'].rolling(window=window).mean().plot(ax=ax)
    ax.set_title('Average fitness')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Average population fitness')

    with tempfile.TemporaryDirectory() as tmpdir:
        plt.savefig(f"{tmpdir}/{file}")
        mlflow.log_artifact(f"{tmpdir}/{file}")


def plot_reward(file, metrics_df, window):
    fig, ax = plt.subplots(figsize=(14, 6))
    metrics_df['reward'].rolling(window=window).mean().plot(ax=ax)
    ax.set_title('Average reward')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Reward')
    with tempfile.TemporaryDirectory() as tmpdir:
        plt.savefig(f"{tmpdir}/{file}")
        mlflow.log_artifact(f"{tmpdir}/{file}")


def plot_classifiers(file, metrics_df, window):
    fig, ax = plt.subplots(figsize=(14, 6))

    metrics_df['population'].rolling(window=window).mean().plot(figsize=(14, 6),
                                                                label='population',
                                                                ax=ax)
    metrics_df['reliable'].rolling(window=window).mean().plot(figsize=(14, 6),
                                                              label='reliable',
                                                              ax=ax)
    plt.legend()
    with tempfile.TemporaryDirectory() as tmpdir:
        plt.savefig(f"{tmpdir}/{file}")
        mlflow.log_artifact(f"{tmpdir}/{file}")


@click.command(help="Perform experiments over Mountain Car environment")
@click.option("--environment", type=click.STRING, default="MountainCar-v0",
              help="Exploration steps")
@click.option("--explore-trials", type=click.INT, default=1000,
              help="Exploration trials")
@click.option("--exploit-trials", type=click.INT, default=10,
              help="Exploitation trials")
@click.option("--position-bins", type=click.INT, default=10)
@click.option("--velocity-bins", type=click.INT, default=20)
@click.option("--biased-exploration-prob", type=click.FLOAT, default=0.00,
              help='Probability of executing biased exploration')
@click.option("--decay", type=click.BOOL, default=False,
              help="Decay in exploration phase")
@click.option("--gamma", type=click.FLOAT, default=0.95)
def run(environment, explore_trials, exploit_trials, position_bins, velocity_bins,
        biased_exploration_prob, decay, gamma):
    bins = [position_bins, velocity_bins]

    with mlflow.start_run():
        logging.info("Initializing environment...")
        env = gym.make(environment)
        env._max_episode_steps = 1000

        logging.info("Creating real-value discretizer...")
        _range, _low = (env.observation_space.high - env.observation_space.low, env.observation_space.low)

        class MountainCarAdapter(EnvironmentAdapter):
            @classmethod
            def to_genotype(cls, obs):
                r = (obs + np.abs(_low)) / _range
                b = (r * bins).astype(int)
                return b.astype(str).tolist()

        logging.info("Creating custom metrics")

        def mc_metrics(pop, env):
            metrics = {'avg_fitness': np.mean([cl.fitness for cl in pop if cl.is_reliable()])}
            metrics.update(population_metrics(pop, env))

            return metrics

        logging.info("Building agent configuration...")
        cfg = Configuration(
            classifier_length=2,
            number_of_possible_actions=3,
            epsilon=1.0,
            biased_exploration=biased_exploration_prob,
            beta=0.2,
            gamma=gamma,
            theta_as=50,
            theta_exp=100,
            theta_ga=50,
            do_ga=True,
            mu=0.03,
            chi=0.0,
            metrics_trial_frequency=5,
            user_metrics_collector_fcn=mc_metrics,
            environment_adapter=MountainCarAdapter)

        trials = int(explore_trials/2)

        logging.info(f"Running {trials} experiments with pure exploration (decay = False)")
        agent = ACS2(cfg)
        population, metrics_1 = agent.explore(env, trials, decay=False)

        logging.info(f"Running {trials} experiments with optional decay (decay = {decay})")
        agent = ACS2(cfg, population)
        population, metrics_2 = agent.explore(env, trials, decay=decay)

        logging.info("Generating metrics...")
        explore_metrics_df = merge_metrics(metrics_1, metrics_2, cfg)
        log_metrics("explore", explore_metrics_df)

        logging.info("Logging population artifact...")
        log_population_artifact("explore-population", population)
        log_metrics_artifact("explore-metrics", explore_metrics_df)

        logging.info("Generating plots...")
        # avg_window = int(trials/100)
        avg_window = 1
        plot_steps_in_trial("explore-steps.png", explore_metrics_df, window=avg_window)
        plot_avg_fitness("explore-fitness.png", explore_metrics_df, window=avg_window)
        plot_reward("explore-reward.png", explore_metrics_df, window=avg_window)
        plot_classifiers("explore-classifiers.png", explore_metrics_df, window=avg_window)

        logging.info(f"Running {exploit_trials} exploit trials")
        exploiter = ACS2(cfg, deepcopy(population))
        population_exploit, metrics_3 = agent.exploit(env, exploit_trials)

        logging.info("Generating metrics")
        exploit_metrics_df = metrics_to_df(metrics_3)
        log_metrics_artifact("exploit-metrics", exploit_metrics_df)
        log_metrics("exploit", exploit_metrics_df)


if __name__ == "__main__":
    run()
