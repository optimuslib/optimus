# OptimUS
[![Documentation Status](https://readthedocs.org/projects/optimuslib/badge/?version=latest)](https://readthedocs.org/projects/optimuslib/)

An open-source Python library for solving 3D acoustic wave propagation.

The OptimUS library provides functionality to simulate acoustic wave propagation in an unbounded domain with multiple scatterers. OptimUS solves the Helmholtz equation in multiple domains with homogeneous material parameters, using a boundary element method (BEM). The library targets general acoustical simulation and has functionality for focused ultrasound in biomedical engineering.


## Installation
The OptimUS library and all dependencies are installed and tested in a Docker container. First, install the docker engine on your machine following the instruction on the [`docker`](https://docs.docker.com/engine/install/) website. Then, pull the docker container by running:

```bash
docker pull optimuslib/optimus:latest
```

To start the container on your machine, run:

```bash
docker run -it -v $(pwd):/home/optimus/localwork --workdir /home/optimus/localwork -p 8888:8888 optimuslib/optimus:latest
```
The output will provide the URL and token to access the Jupyter notebook interface from a web browser.

Upon accessing Jupyter, you can execute the notebooks available in the `notebook` directory on this GitHub page.

If you want to get a bash terminal within the container, you can either launch one through the [Jupyter notebook interface](http://localhost:8888) or via Docker as:

```bash
docker run -it --rm -v $(pwd):/home/optimus/localwork --workdir /home/optimus/localwork optimuslib/optimus:latest 
```
In the terminal, you can execute your Python files by running:

```bash
python3 <file_name.py>
```

*Note: depending on the configuration of your machine's OS, you may need to run the above Docker commands as a super user: `sudo docker`.*

## Documentation
Examples are available in the `notebook` directory on this GitHub page. Automatically generated documentation of the Python API
can be found in [Read the Docs optimus project](https://readthedocs.org/projects/optimuslib/).

## Getting help
Questions about the library and its use can be directed to [optimuslib slack channel]().
Errors in the library should be added to the [GitHub issue tracker](https://github.com/optimuslib/optimus/issues).

## Licence
OptimUS is licensed under an MIT licence. Full text of the licence can be found [here](LICENSE.md).

## References
The main references describing the BEM formulations and preconditioners implemented in OptimUS are as follows:

> [1] Haqshenas, S. R., Gélat, P., van 't Wout, E., Betcke, T., & Saffari, N. (2021). A fast full-wave solver for calculating ultrasound propagation in the body. Ultrasonics, 110, 106240. doi:10.1016/j.ultras.2020.106240

> [2] van 't Wout, E., Haqshenas, S. R., Gélat, P., Betcke, T., & Saffari, N. (2021). Benchmarking preconditioned boundary integral formulations for acoustics. International Journal for Numerical Methods in Engineering, nme.6777. doi:10.1002/nme.6777

> [3] van 't Wout, E., Haqshenas, S. R., Gélat, P., Betcke, T., & Saffari, N. (2022). Boundary integral formulations for acoustic modelling of high-contrast media. Computers & Mathematics with Applications, 105, 136-149. doi:10.1016/j.camwa.2021.11.021

> [4] van 't Wout, E., Haqshenas, S. R., Gélat, P., Betcke, T., & Saffari, N. (2022). Frequency-robust preconditioning of boundary integral equations for acoustic transmission. Journal of Computational Physics, 111229. doi:10.1016/j.jcp.2022.111229


## Citation
If you use OptimUS in your work, please cite us as [follows](CITATION.cff):

**APA**
```
Gélat, P., Haqshenas, S. R., and van 't Wout, E. (2022), OptimUS: A Python library for solving 3D acoustic wave propagation, https://github.com/optimuslib/optimus
```

**BibTeX**
```
@software{optimuslib,
author = {Gélat, Pierre and Haqshenas, Reza and van 't Wout, Elwin},
title = {OptimUS},
url = {https://github.com/optimuslib/optimus},
version = {0.1.0}
}
```


## Acknowledgement
- The OptimUS library uses the bempp-legacy library from the [BEMPP project](https://github.com/bempp) as the computational backend. 
- The tissue properties database is based on [Tissue Properties Database V4.1 of IT'IS Foundation](https://itis.swiss/virtual-population/tissue-properties/downloads/database-v4-1/).


