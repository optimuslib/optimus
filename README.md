# OptimUS
A Python library for solving 3D acoustic wave propagation.

## Description
The OptimUS library provides functionality to simulate acoustic wave propagation in an unbounded domain with multiple scatterers.
The Helmholtz equation models propagation in multiple domains with homogeneous material parameters.
This model is solved with a boundary element method (BEM).
The library targets general acoustical simulation and has functionality for focused ultrasound in biomedical engineering.


## Installation
The OptimUS library can only be used in a Docker image.

## Acknowledgement
The OptimUS library uses the BEMPP project as computational backend, see https://github.com/bempp.
Currently, only version 3 of BEMPP is supported.
Future releases of OptimUS will include the bempp-cl version.
