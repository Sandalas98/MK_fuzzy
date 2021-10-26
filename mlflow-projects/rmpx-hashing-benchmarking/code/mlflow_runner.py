from os import environ
import mlflow
import click

from train import run

if "MLFLOW_TRACKING_URI" not in environ:
    mlflow.set_tracking_uri("http://localhost/mlflow")


@click.command()
@click.option("--experiment", type=click.INT, default=0)
@click.option("--rmpx-size", type=click.Choice(['3', '6', '11', '20', '37']),
              required=True)
@click.option("--trials", type=click.INT, default=100)
@click.option("--modulo", type=click.INT, default=16)
@click.option("--hash",
              type=click.Choice(['SHA256', 'MD5'], case_sensitive=False),
              required=True)
@click.option("--agent",
              type=click.Choice(['ACS', 'ACS2', 'ACS2GA', 'YACS'],
                                case_sensitive=False),
              required=True)
def execute(experiment, rmpx_size, trials, modulo, hash, agent):
    rmpx_size = int(rmpx_size)
    mlflow.set_tag('mode', 'lightweight')
    mlflow.set_tag('experiment', experiment)

    run(rmpx_size, trials, agent, hash, modulo)


if __name__ == '__main__':
    execute()
