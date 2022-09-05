#FROM bempp/notebook:numba
### srhaqshenas/optimus:base is identical to optimus:latest on WS1
FROM optimuslib/optimus:notebook

# RUN export DEBIAN_FRONTEND=noninteractive && \
#     apt-get -qq update && \
#     apt-get -yq --with-new-pkgs -o Dpkg::Options::="--force-confold" upgrade && \
#     apt-get -y install \
#     pkg-config \
#     python3-dev \
#     python3-matplotlib \
#     python3-tk
#     python3-numpy \
#     python3-pip \
#     python3-scipy \
#     python3-setuptools && \
#     apt-get -y install \
#     doxygen \
#     git \
#     graphviz \
#     sudo \
#     valgrind \
#     wget && \
#     apt-get -y install \
#     libglu1 \
#     libxcursor-dev \
#     libxinerama1 && \
#     apt-get -y install \
#     python3-lxml && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Un/Install Python packages (via pip)
# RUN pip3 uninstall --no-cache-dir -y gmsh  
# RUN pip install --no-cache-dir  --upgrade pip setuptools
 
## the below is necessary to include the latest version of matplotlib with all libraries
RUN pip3 install --upgrade matplotlib --ignore-installed six
RUN pip3 install ipywidgets
## the versions of ipympl and matplotlib must be compatible.
RUN pip3 install ipympl==0.5.0

RUN pip3 install --no-cache-dir k3d joblib
# RUN pip3 install --no-cache-dir vtk==8.1.2 
# RUN pip3 install --no-cache-dir scooby==0.5.0
# RUN pip3 install --no-cache-dir pyvista==0.24.0

### to install optimus package from the local drive, uncomment the below 2 lines and amend the path and the name of the whl file
COPY /dist/*.whl /tmp/optimus-0.1.0-py3-none-any.whl
RUN pip3 install --no-cache-dir /tmp/optimus-0.1.0-py3-none-any.whl

### to install optimus package from deployed testpypi website, uncomment and use below command
# RUN pip3 install --extra-index-url https://test.pypi.org/simple/ optimus

