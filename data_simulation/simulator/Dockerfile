FROM python:3.9-slim-buster

WORKDIR /app

COPY simulator/requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

COPY simulator/script.py /app/script.py

COPY ./data ./data

CMD ["python", "script.py"]