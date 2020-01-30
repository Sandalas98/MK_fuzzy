FROM python:3.7

RUN apt-get update && apt-get install -y swig cmake

ADD requirements.txt requirements.txt
ADD requirements-docker.txt requirements-docker.txt

RUN pip install -r requirements.txt
RUN pip install -r requirements-docker.txt

CMD ["jupyter", "lab", "--port", "9998", "--ip=0.0.0.0", "--allow-root"]
