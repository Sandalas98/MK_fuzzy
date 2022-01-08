# Experiments with MLFlow

Create a local `.env` file using the template `.env.template`. Then:

    docker-compose -f docker-compose.yml up

To remove the whole infrastructure: (add `-v` option to delete DB)

    docker-compose down

In the browser go to [localhost/mlflow/](localhost/mlflow/).

# Work scheduling

Python RQ framework is used as a simple work orchestrator.

    pip install 
