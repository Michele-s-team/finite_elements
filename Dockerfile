FROM quay.io/fenicsproject/stable:latest
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    libglu1 \
    libxcursor-dev \
    libxinerama1 \
    libgl-dev \
    libxft2 \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install scipy
RUN pip3 install progressbar
RUN pip3 install meshio
RUN pip3 install gmsh
RUN pip3 install pygmsh
RUN pip3 install h5py

  
