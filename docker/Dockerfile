FROM ubuntu:14.04
MAINTAINER Luke Jaffe <lukejaffe1@gmail.com>

# Install basic dependencies
RUN apt-get update
RUN apt-get install -y build-essential

# Install python and pip
RUN apt-get install -y libpython-dev
RUN apt-get install -y python-pip

# Install other packages
RUN apt-get install -y vim

# Install python packages
RUN apt-get install -y python-numpy
RUN apt-get install -y python-scipy
RUN apt-get install -y python-matplotlib

# Install eog
RUN apt-get install -y eog

# Install ipython
RUN pip install setuptools==33.1.1
RUN pip install ipython

# Install python libs
RUN pip install progressbar2

# Install git
RUN apt-get install -y wget git

# Install pypy
WORKDIR /opt
RUN wget https://bitbucket.org/pypy/pypy/downloads/pypy2-v5.6.0-linux64.tar.bz2
RUN tar jxf pypy2-v5.6.0-linux64.tar.bz2 
RUN ln -s `pwd`/pypy2-v5.6.0-linux64/bin/pypy /usr/bin/pypy

# Install pip
WORKDIR /opt
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN pypy get-pip.py

# Install pypy packages
RUN pypy -m pip install numpy
RUN apt-get install -y libfreetype6-dev
RUN pypy -m pip install freetype-py
RUN apt-get install -y pkg-config
RUN pypy -m pip install matplotlib

# Install scipy
RUN apt-get install -y gfortran libopenblas-dev liblapack-dev python-dev
RUN pip install scipy --upgrade

# Install networkx
RUN pip install networkx

# Install jupyter
RUN pip install jupyter

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
