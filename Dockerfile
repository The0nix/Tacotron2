FROM pure/python:3.8-cuda10.2-base

WORKDIR /home/user

COPY requirements.txt .

RUN apt-get -y update && apt-get -y install libsndfile1-dev

RUN pip install -r requirements.txt

ENV PYTHONPATH=src

COPY src src

CMD python ./src/fit_label_encoder.py && python ./src/train.py
