FROM python:3.7.13-slim-buster
RUN mkdir /opt/fast
WORKDIR /opt/fast
COPY requirements.txt .
RUN pip install -r requirements.txt 
COPY . .
# https://stackoverflow.com/questions/43764624/importerror-libgomp-so-1-cannot-open-shared-object-file-no-such-file-or-direc
RUN apt-get update && apt-get install libgomp1
CMD ["streamlit", "run", "streamlite.py"]