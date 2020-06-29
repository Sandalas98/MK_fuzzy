from lcs.agents import EnvironmentAdapter
from lcs.metrics import population_metrics


class Adapter(EnvironmentAdapter):

    @classmethod
    def to_genotype(cls, phenotype):
        # Represent state as a single unicode character
        return chr(int(phenotype) + 65)


def fsw_metrics(agent, env):
    pop = agent.get_population()

    metrics = {
        'reliable': len([cl for cl in pop if cl.is_reliable()]),
    }

    if agent.rho is not None:
        metrics['rho'] = agent.rho

    metrics.update(population_metrics(pop, env))
    return metrics
