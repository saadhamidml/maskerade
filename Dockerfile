# devel is necessary since pykeops depends on nvcc.
FROM  pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel

# Supress prompts to choose location during R install. Install git to log information to Sacred.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
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
  pykeops==1.4.1 \
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

# Create a user. Ensure you replace the UID and GID with your user's.
ARG USERNAME=maskerade
ARG USER_UID=1007
ARG USER_GID=1008
# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
# Set the default user. Omit if you want to keep the default as root.
USER $USERNAME

RUN pip install tensorboard

# ENTRYPOINT [ "/bin/bash" ]
# ENTRYPOINT  ["python", "main.py", "--force", "with"]
# CMD ["config/yacht/bq.yaml"]
# ENTRYPOINT ["bash", "./repeat_runs.sh"]
# CMD ["-c config/yacht/bq.yaml", "-r 10"]

# Do (from project root directory in host i.e. .../maskerade):
# docker build -t maskerade .
# docker run -it -u $(id -u):$(id -g) -v $(pwd):/maskerade --gpus=all --network=mongo-network maskerade
