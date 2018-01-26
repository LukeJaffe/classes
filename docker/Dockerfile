FROM ubuntu:14.04
MAINTAINER Luke Jaffe <lukejaffe1@gmail.com>

# Install basic dependencies
RUN apt-get update
RUN apt-get install -y build-essential

# Install other packages
RUN apt-get install -y vim

# Install eog
RUN apt-get install -y eog

# Install git
RUN apt-get install -y wget git

# Install pypy packages
RUN apt-get install -y libfreetype6-dev
RUN apt-get install -y pkg-config

# Install scipy
RUN apt-get install -y gfortran libopenblas-dev liblapack-dev

# Install python3
RUN apt-get install -y libpython3-dev
RUN apt-get install -y python3-pip
RUN apt-get install -y python3-numpy
RUN apt-get install -y python3-scipy
RUN apt-get update
RUN apt-get install -y python3-matplotlib

# Install python3 libs
RUN pip3 install setuptools==33.1.1
RUN pip3 install pyparsing --upgrade
RUN pip3 install ipython
RUN pip3 install progressbar2
RUN pip3 install networkx
RUN pip3 install jupyter
RUN pip3 install Cython

# Set user specific env
ENV HOST_UID=1001

# Create user 'username'
RUN useradd --create-home --shell /bin/bash --groups sudo username
RUN echo "username:password" | chpasswd

# Change id of username
RUN usermod -u $HOST_UID username
RUN groupmod -g $HOST_UID username
