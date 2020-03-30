<<<<<<< HEAD
## ACS experiments

All experiments are based on MlFlow framework.

     mlflow run mountaincar -P environment=EnergyMountainCar-v0

### UI

    mlflow ui

## Docker

Build Docker image with PyALCS implementation (currnt github) an and Jupyter Lab image   

    docker build -f Dockerfile -t acs . 
    
You can run it locally to test if everything is working fine

    docker run --rm --name jupyter -p 9998:9998 -v `pwd`/notebooks:/notebooks acs

Finally push to docker hub
    
    docker login
    docker tag acs acs/notebook
    docker push acs/notebook

## Development

    conda env create --file conda.yaml 
    conda activate pyalcs-experiments

    pip install -r requirements.txt
    pip install -f /Users/khozzy/Projects/pyalcs -f /Users/khozzy/Projects/openai-envs -r requirements-dev.txt
    
    make test
