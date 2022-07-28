FROM python:3.9-slim



RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip



COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
RUN python -m pip install --user -rrequirements.txt

COPY --chown=algorithm:algorithm process.py /opt/algorithm/
COPY --chown=algorithm:algorithm model_late.py /opt/algorithm/
COPY --chown=algorithm:algorithm unet_late_model1.h5 /opt/algorithm/
COPY --chown=algorithm:algorithm unet_late_model2.h5 /opt/algorithm/
COPY --chown=algorithm:algorithm unet_late_model3.h5 /opt/algorithm/
COPY --chown=algorithm:algorithm unet_late_model4.h5 /opt/algorithm/
COPY --chown=algorithm:algorithm unet_late_model5.h5 /opt/algorithm/

ENTRYPOINT python -m process $0 $@
