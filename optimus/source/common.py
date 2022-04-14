"""Common functionality for acoustic sources."""

import bempp.api as _bempp


class Source:
    def __init__(
        self,
        source_type,
        frequency,
    ):
        """
        The source of the acoustic wave.
        """

        if not isinstance(frequency, (int, float)):
            raise TypeError("Frequency should be specified as a number")
        elif frequency <= 0.0:
            raise ValueError("Frequency should have a positive value.")

        self.type = source_type
        self.frequency = frequency

    def pressure_field(self, locations, medium):
        """
        Calculate the pressure field in the specified locations.
        Needs to be overridden by specific source type.
        """
        raise NotImplementedError

    def normal_pressure_gradient(self, locations, normals, medium):
        """
        Calculate the normal gradient of the pressure field in the
         specified locations.
        Needs to be overridden by specific source type.
        """
        raise NotImplementedError

    def calc_surface_traces(self, medium, spaces):
        """
        Calculate the Dirichlet and Neumann traces of the source on a
         surface grid, where the incident field is given by explicit
         functions for the pressure and its normal gradient.

        Parameters
        ----------
        medium : optimus.material.Material
            The propagating medium.
        spaces : tuple of bempp.api.FunctionSpace
            The discrete spaces on the surface grid.

        Returns
        ----------
        traces : tuple of bempp.api.GridFunctions
            The Dirichlet and Neumann traces.
        """

        def dirichlet_fun(x, n, domain_index, result):
            result[0] = self.pressure_field(x, medium)

        def neumann_fun(x, n, domain_index, result):
            result[0] = self.normal_pressure_gradient(x, n, medium)

        space_dirichlet = spaces[0]
        trace_dirichlet = _bempp.GridFunction(
            space_dirichlet, fun=dirichlet_fun
        )

        space_neumann = spaces[0]
        trace_neumann = _bempp.GridFunction(space_neumann, fun=neumann_fun)

        return trace_dirichlet, trace_neumann
