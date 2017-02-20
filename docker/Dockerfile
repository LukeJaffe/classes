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

# Create user 'username'
RUN useradd --create-home --shell /bin/bash --groups sudo username
RUN echo "username:password" | chpasswd

