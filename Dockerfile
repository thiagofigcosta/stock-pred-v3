FROM python:3.9 AS builder

RUN apt-get install -y --no-install-recommends make cmake gcc lib6-dev

RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz /tmp/ta_lib.gz
RUN tar -xzf /tmp/ta_lib.gz
RUN cd /tmp/ta-lib/ && ./configure && make && make install

COPY requirements_no_ver.txt .
RUN pip install --user -r requirements.txt



FROM python:3.9-slim AS slim
WORKDIR /code

RUN apt-get install -y --no-install-recommends graphviz make cmake gcc lib6-dev procps

COPY --from=builder /tmp/ta_lib.gz /tmp/ta_lib.gz
RUN tar -xzf /tmp/ta_lib.gz
RUN cd /tmp/ta-lib/ && ./configure && make && make install

COPY --from=builder /root/.local /root/.local
COPY entrypoint.sh .
COPY clean_up_generated_files.sh .
COPY ./*.py ./
COPY datasets/. datasets/
RUN mkdir -p hyperparameters ; mkdir -p logs ; mkdir -p models ; mkdir -p nas ; \
mkdir -p prophets ; mkdir -p saved_plots ; mkdir -p experiments

RUN chmod +x entrypoint.sh
RUN chmod -R 777 hyperparameters
RUN chmod -R 777 logs
RUN chmod -R 777 models
RUN chmod -R 777 nas
RUN chmod -R 777 prophets
RUN chmod -R 777 saved_plots
RUN chmod -R 777 experiments

ENV PATH=/root/.local/bin:$PATH

CMD ["entrypoint.sh"]