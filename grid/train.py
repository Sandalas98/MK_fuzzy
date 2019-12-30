import tempfile
from copy import deepcopy

import numpy as np
import click
import mlflow
import gym
import gym_grid

from lcs.agents.acs2 import Configuration, ACS2


def log_population(filename, population):
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(f"{tmpdir}/{filename}", mode="w") as f:

            # Dump data
            for cl in sorted(population, key=lambda c: -c.fitness):
                f.write(f"{cl}\n")

            f.flush()
            mlflow.log_artifact(f.name)


def log_average_steps(param, metrics):
    steps = [m['steps_in_trial'] for m in metrics]
    avg_steps = np.mean(steps)
    mlflow.log_metric(param, avg_steps)


@click.command(help="Perform experiments over Grid environment")
@click.option("--environment", type=click.STRING, default="grid-10-v0",
              help="Exploration steps")
@click.option("--explore-steps", type=click.INT, default=5,
              help="Exploration steps")
@click.option("--exploit-steps", type=click.INT, default=1,
              help="Exploitation steps")
@click.option("--decay", type=click.BOOL, default=False,
              help="Decay in exploration phase")
def run(environment, explore_steps, exploit_steps, decay):
    with mlflow.start_run():
        # Initialize environment
        env = gym.make(environment)

        # Create experiment configuration
        cfg = Configuration(
            classifier_length=2,
            number_of_possible_actions=4,
            epsilon=0.9,
            beta=0.03,
            gamma=0.97,
            theta_i=0.1,
            theta_as=10,
            theta_exp=50,
            do_ga=True,
            mu=0.04,
            u_max=10,
            metrics_trial_frequency=10)

        # Start the experiment
        explorer = ACS2(cfg)
        population_explore, metrics_explore = explorer.explore(env, explore_steps, decay=decay)

        exploiter = ACS2(cfg, deepcopy(population_explore))
        population_exploit, metrics_exploit = exploiter.exploit(env, exploit_steps)

        # Log data
        log_population("population_explore.txt", population_explore)
        log_population("population_exploit.txt", population_exploit)

        log_average_steps("explore_avg_steps", metrics_explore)
        log_average_steps("exploit_avg_steps", metrics_exploit)

        for m in metrics_explore:
            mlflow.log_metric("explore_steps", m['steps_in_trial'], m['trial'])

        for m in metrics_exploit:
            mlflow.log_metric("exploit_steps", m['steps_in_trial'], m['trial'])


if __name__ == "__main__":
    run()
