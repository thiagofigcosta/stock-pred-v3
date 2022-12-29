# stock-pred-v3, the prophet of the stock market

Based on: https://https://github.com/thiagofigcosta/stock-pred-v2

## Running on Host

### Virtual environment:

To set up virtual env

```
pip3 install virtualenv
python3 -m virtualenv -p python3 venv
```

To activate virtual env

```
source venv/bin/activate
```

Install dependencies and run the application on the activated enviroment!

To deactivate virtual env

```
deactivate
```

To delete venv:

```
rm -rf venv
```

### Install deps linux:

Install dependencies:

```
pip3 install -r requirements.txt
```

Install graphviz

```
sudo yum install graphviz
```

or

```
sudo apt install graphviz
```

or

```
sudo zypper install graphviz
```

Install [TA-Lib](https://pypi.org/project/TA-Lib/).

### Install deps mac:

```
brew install graphviz
brew install python@3.9
CPPFLAGS="$CPPFLAGS $(python3.9-config --include)"
brew install ta-lib
pip install TA-Lib
```

### Install deps windows:

Open anaconda console and type:

```
pip install pydot
pip install numpy
pip install keras
pip install --user tensorflow
```

Install [graphviz](https://graphviz.gitlab.io/download/).

Install [TA-Lib](https://pypi.org/project/TA-Lib/).

## Running with Docker

### To build image

```
docker build -t stock-pred:3.0.0 .
```

### To run image de-attached

```
docker run -d stock-pred:3.0.0
```

or

```
docker run -e MODE='lstm' -d stock-pred:3.0.0
```

## Running with Docker-Compose

### To build

```
docker-compose build
```

### To Run

```
docker-compose up -d
```

## Useful Docker commands

### To follow logs

```
docker logs --follow $(docker container ls | grep stock-pred | cut -f 1 -d' ' | head -n 1)
```

### To access running container

```
docker exec -it $(docker container ls | grep stock-pred | cut -f 1 -d' ' | head -n 1) bash
```

### To stop running container

```
docker stop $(docker container ls | grep stock-pred | cut -f 1 -d' ' | head -n 1)
```

### To copy experiment results compressed

```
docker exec -it $(docker container ls | grep stock-pred | cut -f 1 -d' ' | head -n 1) bash -c "tar -zcvf /code/experiments/exp.tar.gz /code/experiments/saved_plots/"
docker cp $(docker container ls | grep stock-pred | cut -f 1 -d' ' | head -n 1):/code/experiments/exp.tar.gz .
```

### To uncompress experiments

```
mkdir -p experiment ; tar -zxvf exp.tar.gz -C experiment
```

## Clean up root owned volumes manually with dummy image

### Create a `cleaner.Dockerfile` file

With the contents:

```Dockerfile
FROM python:3.9-slim AS slim
RUN printf '#!/bin/bash\n\ntail -f /dev/null\n' > /entrypoint.sh

RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
```

### Create a `cleaner-docker-compose.yml` file

With the contents, be sure to edit the volumes section with your needs:

```yml
version: "3.3"
services:
  docker-root-owned-cleaner:
    build:
      context: .
      dockerfile: cleaner.Dockerfile
    restart: always
    volumes:
      - ./docker_volumes:/docker_volumes
```

### Run the dummy image

```bash
docker-compose build -f cleaner-docker-compose.yml
docker-compose up -d -f cleaner-docker-compose.yml
```

### Clean it up

Do it manually:

```bash
docker exec -it $(docker container ls | grep docker-root-owned-cleaner | cut -f 1 -d' ' | head -n 1) bash
```
