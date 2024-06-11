ARG BASE_IMAGE="docker-hub.bitdefender.net:443/snapshot/fgogianu-big-guns-f56d7f0d:06-01-19_09-43"

FROM $BASE_IMAGE

RUN mkdir /build
WORKDIR /build
COPY . .

ENTRYPOINT [ "/bin/bash", "-c" ]
CMD [ "source activate torch110 && liftoff /build/evaluation_problem.py /data/results/theodon-tabular/2019May31-172609_ens_b_alpha_sweep/ --procs-no 4  --per-gpu 4" ]