FROM python:3.12-bookworm

ARG WD_USER=administrador
ARG WD_UID=1000
ARG WD_GROUP=administrador
ARG WD_GID=1000


# We rarely see a full upgrade in a Dockerfile. Why?
# && apt-get --assume-yes dist-upgrade \
RUN apt-get update \
  && apt-get --assume-yes --no-install-recommends install \
  cython3 \
  ffmpeg \
  unzip \
  wget \
  vim \
  git
RUN rm -rf /var/lib/apt/lists/*

RUN addgroup --gid $WD_GID $WD_GROUP \
  && adduser --uid $WD_UID --gid $WD_GID --shell /bin/bash --no-create-home $WD_USER \
  && chown -R $WD_USER:$WD_GROUP /home

USER $WD_USER:$WD_GROUP

RUN mkdir /home/$WD_USER
WORKDIR /home/$WD_USER

COPY nemo_msdd_configs/ nemo_msdd_configs/
COPY utils.py .
COPY helpers.py .
COPY main.py .
COPY requirements.txt .

RUN mkdir venv \
	&& python -m venv venv \
	&& . venv/bin/activate \
	&& pip install -U --no-cache-dir pip \
	#&& pip install -U --no-cache-dir setuptools \
	#&& pip install -U --no-cache-dir wheel \
	#&& pip install --upgrade --no-cache-dir numpy==1.26.4 \
	#&& pip install Cython \
	&& pip install --no-cache-dir --requirement requirements.txt

ENV LD_LIBRARY_PATH=/home/administrador/venv/lib/python3.12/site-packages/nvidia/cudnn/lib

EXPOSE 8080

CMD ["venv/bin/fastapi","run", "main.py", "--port", "8080"]