FROM quay.io/fenicsproject/stable:current
USER root
RUN apt-get -qq update && \
    apt-get -y upgrade && \
    apt-get clean && \
    apt-get -y install python3-h5py \
    libglu1 \
    libxcursor-dev \
    libxinerama1 \
    libgl-dev \
    libxft2 && \
    pip install --upgrade pip && \
    pip install meshio[all] && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
USER root

RUN pip3 install colorama
RUN pip3 install shapely
RUN pip3 install scipy
RUN pip3 install progressbar
RUN pip3 install meshio
RUN pip3 install gmsh
RUN pip3 install pygmsh
RUN pip3 install h5py