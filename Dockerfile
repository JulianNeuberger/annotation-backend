FROM python:3.9

WORKDIR /app
ADD ./requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
ADD . /app

RUN python -m spacy download en_core_web_sm

EXPOSE 5000
ENTRYPOINT FLASK_APP=/app/api/api.py flask run --host=0.0.0.0 --port=5000