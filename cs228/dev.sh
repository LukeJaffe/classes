#!/bin/bash

XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth-n
xauth nlist | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

docker run -it --rm \
 --name hw \
 -v ${HOME}/.vimrc:/home/username/.vimrc \
 -v ${HOME}/.vim:/home/username/.vim \
 -v `pwd`:/home/username/work \
 -v $XSOCK:$XSOCK \
 -v $XAUTH:$XAUTH \
 -e XAUTHORITY=$XAUTH \
 -e DISPLAY=$DISPLAY \
 -e QT_X11_NO_MITSHM=1 \
 -e QT_CRASH_OUTPUT=/home/username/qt_crash.log \
 --user=username \
 -w /home/username/work/cs228/hw/5/starter \
 --privileged \
 classes:python /bin/bash
