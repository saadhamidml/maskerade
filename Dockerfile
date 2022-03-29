# devel is necessary since pykeops depends on nvcc.
FROM  pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel

# Supress prompts to choose location during R install. Install git to log information to Sacred.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends r-base
RUN apt-get install -y build-essential libssl-dev wget git && \
  wget https://github.com/Kitware/CMake/releases/download/v3.19.1/cmake-3.19.1.tar.gz && \
  tar -zxvf cmake-3.19.1.tar.gz && \
  cd cmake-3.19.1 && \
  ./bootstrap && \
  make && \
  make install
ENV R_HOME=/usr/lib/R
RUN pip install gpytorch==1.3.0 \
  botorch==0.3.3 \
  ax-platform==0.1.19 \
  pyro-ppl==1.5.1 \
  matplotlib \
  multimethod \
  pykeops \
  pymongo \
  pyyaml \
  quandl \
  requests \
  rpy2 \
  sacred \
  tqdm
RUN git clone https://github.com/wjmaddox/spectralgp.git
RUN mv /workspace/spectralgp/spectralgp /opt/conda/lib/python3.8/site-packages/spectralgp

# Uncomment below to copy project files into container and run command when container starts up.
#COPY ./maskerade /maskerade
#
# If project files will change a lot it is better to use a bind-mount.
# In this case uncomment below.
RUN mkdir -p /maskerade/src

ENV PYTHONPATH=/maskerade/src
WORKDIR /maskerade/src

# Do (from project root directory in host i.e. .../maskerade):
# docker build -t maskerade .
# docker run -it -u $(id -u):$(id -g) -v $(pwd):/maskerade --gpus=all maskerade
