**OptimUS** documentation
===================================

An open-source Python library for solving 3D acoustic wave propagation.

The OptimUS library provides functionality to simulate acoustic wave propagation in an unbounded domain with multiple scatterers. OptimUS solves the Helmholtz equation in multiple domains with homogeneous material parameters, using a boundary element method (BEM). The library targets general acoustical simulation and has functionality for focused ultrasound in biomedical engineering.


**Installation**
===================================
The OptimUS library and all dependencies are installed and tested in a Docker container. First, install the docker engine on your machine following the instruction on the `docker <https://docs.docker.com/engine/install/>`_ website. Then, pull the docker container by running:

```
docker pull optimuslib/optimus:latest
```


To start the container on your machine, run:


```
docker run -it -v $(pwd):/home/optimus/localwork --workdir /home/optimus/localwork -p 8888:8888 optimuslib/optimus:latest
```

The output will provide the URL and token to access the Jupyter notebook interface from a web browser.

Upon accessing Jupyter, you can execute the notebooks available in the *notebook* directory on the `GitHub page <https://github.com/optimuslib/optimus>`_.

If you want to get a bash terminal within the container, you can either launch one through the *Jupyter notebook interface* or via Docker as:


```
docker run -it --rm -v $(pwd):/home/optimus/localwork --workdir /home/optimus/localwork optimuslib/optimus:latest 
```

In the terminal, you can execute your Python files by running:


```
python3 <file_name.py>
```


.. note::

   This project is under active development.


Contents
--------

.. toctree::

   Home <self>

