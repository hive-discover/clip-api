FROM python:3.7-slim-buster
   
WORKDIR /app
EXPOSE 8080

RUN apt-get update &&\
    apt-get install --no-install-recommends --yes build-essential libgl1 libglib2.0-0 libsm6 libxrender1 libxext6

COPY data/ data/

# Install Requirements (cache it)
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# Download default CLIP Model (cache it)
COPY models.py models.py
RUN python3 models.py

# Copy the rest of the code
COPY . .


ENTRYPOINT ["python3"]
CMD ["entrypoint.py"] 

# CMD could also be api.py or worker.py
# docker run --env-file V:\Projekte\HiveDiscover\Python\docker_variables.env registry.hive-discover.tech/clip-api:0.7