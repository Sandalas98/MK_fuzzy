import logging
import os
from lcs.agents.acs import ACS, Configuration

import gym  # noqa: E402
# noinspection PyUnresolvedReferences
import gym_grid  # noqa: E402


# Configure logger
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def print_cl(cl):
    action = None
    if cl.action == 0:
        action = '⬅'
    if cl.action == 1:
        action = '➡'
    if cl.action == 2:
        action = '⬆'
    if cl.action == 3:
        action = '⬇'
    print(f"{cl.condition} - {action} - {cl.effect} "
          f"{'(' + str(cl.mark) + ')':21} "
          f"[fit: {cl.fitness:.3f}, r: {cl.r:.2f}, q: {cl.q:.2f}]")


if __name__ == '__main__':
    # Load desired environment
    grid = gym.make('grid-10-v0')

    # Configure and create the agent
    cfg = Configuration(
        classifier_length=2,
        number_of_possible_actions=4,
        epsilon=0.95,
        beta=0.01,
        theta_i=0.1,
        metrics_trial_frequency=10)

    # Explore the environment
    agent = ACS(cfg)
    population, explore_metrics = agent.explore(grid, 500)

    for cl in sorted(population, key=lambda c: -c.fitness):
        print_cl(cl)

    # Exploit
    agent2 = ACS(cfg, population)
    pop_exploit, metric_exploit = agent2.exploit(grid, 100)
