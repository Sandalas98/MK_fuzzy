import logging

import gym  # noqa: E402
# noinspection PyUnresolvedReferences
import gym_grid  # noqa: E402
from lcs import Perception
from lcs.agents import EnvironmentAdapter
from lcs.agents.acs2 import ACS2, Configuration
from lcs.metrics import population_metrics
from lcs.strategies.action_selection2 import BestAction

# Configure logger
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


ACTIONS =  ['⬅', '➡', '⬆', '⬇']


# Define grid size
grid_size = 5

def grid_transitions(grid_size):
    MAX_POS = grid_size
    LEFT, RIGHT, UP, DOWN = 0, 1, 2, 3

    def _handle_state(state):
        moves = []
        (x, y) = state

        # handle inner rectangle - 4 actions available
        if 1 < x < MAX_POS and 1 < y < MAX_POS:
            moves.append(((x, y), LEFT, (x - 1, y)))
            moves.append(((x, y), RIGHT, (x + 1, y)))
            moves.append(((x, y), UP, (x, y + 1)))
            moves.append(((x, y), DOWN, (x, y - 1)))

        # handle bounds (except corners) - 3 actions available
        if x == 1 and y not in [1, MAX_POS]:  # left bound
            moves.append(((x, y), RIGHT, (x + 1, y)))
            moves.append(((x, y), UP, (x, y + 1)))
            moves.append(((x, y), DOWN, (x, y - 1)))

        if x == MAX_POS and y not in [1, MAX_POS]:  # right bound
            moves.append(((x, y), LEFT, (x - 1, y)))
            moves.append(((x, y), UP, (x, y + 1)))
            moves.append(((x, y), DOWN, (x, y - 1)))

        if x not in [1, MAX_POS] and y == 1:  # lower bound
            moves.append(((x, y), LEFT, (x - 1, y)))
            moves.append(((x, y), RIGHT, (x + 1, y)))
            moves.append(((x, y), UP, (x, y + 1)))

        if x not in [1, MAX_POS] and y == MAX_POS:  # upper bound
            moves.append(((x, y), LEFT, (x - 1, y)))
            moves.append(((x, y), RIGHT, (x + 1, y)))
            moves.append(((x, y), DOWN, (x, y - 1)))

        # handle corners - 2 actions available
        if x == 1 and y == 1:  # left-down
            moves.append(((x, y), RIGHT, (x + 1, y)))
            moves.append(((x, y), UP, (x, y + 1)))

        if x == 1 and y == MAX_POS:  # left-up
            moves.append(((x, y), RIGHT, (x + 1, y)))
            moves.append(((x, y), DOWN, (x, y - 1)))

        if x == MAX_POS and y == 1:  # right-down
            moves.append(((x, y), LEFT, (x - 1, y)))
            moves.append(((x, y), UP, (x, y + 1)))

        return moves

    transitions = []
    for x in range(1, MAX_POS + 1):
        for y in range(1, MAX_POS + 1):
            transitions += _handle_state((x, y))

    return transitions

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
          f"[fit: {cl.fitness:.3f}, r: {cl.r:.2f}, ir: {cl.ir:.2f}]")

def calculate_knowledge(population, environment):
    reliable = [c for c in population if c.is_reliable()]
    nr_correct = 0

    mappings = {}

    for start, action, end in transitions:
        mappings[(start, action, end)] = []

    for start, action, end in transitions:
        p0 = Perception([str(el) for el in start])
        p1 = Perception([str(el) for el in end])

        if any([True for cl in reliable if
                cl.predicts_successfully(p0, action, p1)]):
            mappings[(start, action, end)] = [cl for cl in reliable if
                cl.predicts_successfully(p0, action, p1)]
            nr_correct += 1

    return nr_correct / len(transitions) * 100.0


def optimal_action_prob(population, environment):
    m = {}

    for state, _, _ in transitions:
        p0 = Perception([str(el) for el in state])
        match_set = population.form_match_set(p0)
        m[state] = BestAction(4)(match_set)

    return sum(1 for a in m.values() if a in [1, 2]) / len(transitions) * 100

transitions = grid_transitions(grid_size)

# Build agent configuration
# Collect additional population metrics
def grid_metrics(pop, env):
    metrics = {
        'knowledge': calculate_knowledge(pop, env),
        'optimal_action_prob': optimal_action_prob(pop, env)
    }
    metrics.update(population_metrics(pop, env))
    return metrics

class GridAdapter(EnvironmentAdapter):
    @staticmethod
    def to_genotype(phenotype):
        return phenotype,

if __name__ == '__main__':
    # Load desired environment
    grid = gym.make(f'grid-{grid_size}-v0')

    # Configure and create the agent
    cfg = Configuration(
        classifier_length=2,
        number_of_possible_actions=4,
        epsilon=0.9,
        beta=0.2,
        gamma=0.95,
        theta_i=0.1,
        theta_as=50,
        theta_exp=50,
        theta_ga=50,
        do_ga=False,
        mu=0.04,
        u_max=2,
        metrics_trial_frequency=5,
        user_metrics_collector_fcn=grid_metrics
    )

    # Explore the environment
    agent1 = ACS2(cfg)
    population, explore_metrics = agent1.explore(grid, 500, decay=False)

    for cl in sorted(population, key=lambda c: -c.fitness):
        if cl.does_anticipate_change():
            print_cl(cl)

    # Exploit
    agent2 = ACS2(cfg, population)
    pop_exploit, metric_exploit = agent2.exploit(grid, 30)

    # Print classifiers
    for cl in sorted(pop_exploit, key=lambda c: -c.fitness):
        if cl.does_anticipate_change():
            print_cl(cl)
