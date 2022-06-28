FROM python:3.7.13-slim-buster
RUN mkdir /opt/fast
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt 