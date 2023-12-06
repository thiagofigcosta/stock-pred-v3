# stock-pred-v3, the prophet of the stock market

Based on: https://https://github.com/thiagofigcosta/stock-pred-v2

## Running on Host

### Virtual environment:

To set up virtual env

```shell
pip3 install virtualenv
python3 -m virtualenv -p python3 venv
```

To activate virtual env

```shell
source venv/bin/activate
```

Install dependencies and run the application on the activated enviroment!

To deactivate virtual env

```shell
deactivate
```

To delete venv:

```shell
rm -rf venv
```

### Install deps linux:

Install dependencies:

```shell
pip3 install -r requirements.txt # or requirements_no_version.txt
```

Install graphviz

```shell
sudo yum install graphviz
```shell

or

```shell
sudo apt install graphviz
```

or

```shell
sudo zypper install graphviz
```

Install [TA-Lib](https://pypi.org/project/TA-Lib/).

### Install deps mac:

```shell
pip3 install -r requirements_no_version_mac.txt


brew install graphviz
brew install python@3.9
CPPFLAGS="$CPPFLAGS $(python3.9-config --include)"
brew install ta-lib
pip install TA-Lib
```

### Install deps windows:

Open anaconda console and type:

```shell
pip install pydot
pip install numpy
pip install keras
pip install --user tensorflow
```

Install [graphviz](https://graphviz.gitlab.io/download/).

Install [TA-Lib](https://pypi.org/project/TA-Lib/).

## Running with Docker

### To build image

```shell
docker build -t stock-pred:3.0.0 .
```

### To delete image

```shell
docker image rm $(docker image ls | grep stock-pred | cut -f 1 -d' ' | head -n 1)
```

### To run image de-attached

```shell
docker run -d stock-pred:3.0.0
```

or

```shell
docker run -e MODE='lstm' -d stock-pred:3.0.0
```

## Running with Docker-Compose

### To build without using cache 

`docker-compose up --build` uses the build cache

```shell
docker-compose build --no-cache
```

### To Run

```shell
docker-compose up -d
```

### To run with GPU support

```shell
docker-compose -f docker-compose-tf2.yml up -d
```

or

```shell
docker-compose -f docker-compose-cuda.yml up -d
```

### To stop and remove containers, networks, volumes, and images created by docker-compose up

```shell
docker-compose down
```

## Useful Docker commands

### To stock all docker-compose containers

```shell
docker-compose down
```

### To follow logs

```shell
docker logs --follow $(docker container ls | grep stock-pred | cut -f 1 -d' ' | head -n 1)
```

### To access running container

```shell
docker exec -it $(docker container ls | grep stock-pred | cut -f 1 -d' ' | head -n 1) bash
```

### To stop running container

```shell
docker stop $(docker container ls | grep stock-pred | cut -f 1 -d' ' | head -n 1)
```

### To copy experiment results compressed

```shell
docker exec -it $(docker container ls | grep stock-pred | cut -f 1 -d' ' | head -n 1) bash -c "tar -zcvf /code/experiments/exp.tar.gz /code/experiments/saved_plots/"
docker cp $(docker container ls | grep stock-pred | cut -f 1 -d' ' | head -n 1):/code/experiments/exp.tar.gz .
```

### To remove dangling images

```shell
docker image prune
```

### To uncompress experiments

```shell
mkdir -p experiment ; tar -zxvf exp.tar.gz -C experiment
```

### To export docker image

```shell
docker save stock-pred-v3_stock-prev-v3-the-prophet | gzip > stock-pred_ubuntu20-04_py3-10_tensorflow13-1-with-avx-and-cuda11-8_docker-img.tar.gz
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

```shell
docker-compose build -f cleaner-docker-compose.yml
docker-compose up -d -f cleaner-docker-compose.yml
```

### Clean it up

Do it manually:

```shell
docker exec -it $(docker container ls | grep docker-root-owned-cleaner | cut -f 1 -d' ' | head -n 1) bash
```

## Test GPU on docker

```shell
docker run -it --gpus all nvidia/cuda:12.2.0-base-ubuntu20.04 nvidia-smi

```


## System monitor

### Glances

Run:
```shell
docker run --rm --gpus all -e TZ="${TZ}" -v /var/run/docker.sock:/var/run/docker.sock:ro -v /run/user/1000/podman/podman.sock:/run/user/1000/podman/podman.sock:ro --pid host --network host -it nicolargo/glances:latest-full
```

or the command bellow an access [127.0.0.1:61208](http://127.0.0.1:61208):
```shell
docker run -d --restart="always" --gpus all -p 61208-61209:61208-61209 -e TZ="${TZ}" -e GLANCES_OPT="-w" -v /var/run/docker.sock:/var/run/docker.sock:ro -v /run/user/1000/podman/podman.sock:/run/user/1000/podman/podman.sock:ro --pid host nicolargo/glances:latest-full
```


### Nvidia Docker Stats

Clonee [nvidia-docker-stats](https://github.com/AllenCellModeling/nvidia-docker-stats), go to the cloned folder and run:
```shell
python3 -m nvidiadockerstats.nvidiadockerstats
```
