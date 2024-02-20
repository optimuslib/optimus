def plot_pressure_field(
    postprocess_obj,
    field="total",
    unit="Pa",
    display_edges=True,
    clim=None,
    file_name=None,
):
    """2D contour plotting of the real part and the magnitude of the pressure field.

    Parameters
    ----------
    postprocess_obj : optimus.postprocess.PostProcess
        Optimus postprocess object that includes the visualisation pressure fields
    field : string
        The pressure field to be plotted.
        The options are:
            1. total, total_field, or total_pressure;
            2. scattered, scattered_pressure, or scattered_field.
            3. incident, incident_pressure, or incident_field.
    unit : string
        The pressure unit for the plot, where the data is assumed to be in Pa.
        The options are: Pa, kPa, MPa and GPa.
    display_edges : boolean
        Display domains edges, i.e. domains and plane interfaces.
    clim : tuple (float, int)
        The color limits: (clim_min, clim_max).
        Must be of the same units as pressure fields.
    file_name : None, str
        The file name to export the figure, if specified.
        The labels 'real' and 'abs' are prepended.

    Returns
    -------
    fig_p_real, fig_p_tot : matplotlib.Figure
        Contour plots of the real and absolute value of the pressure field.
    """

    import numpy as _np
    import copy as _copy

    plane_axes = postprocess_obj.plane_axes
    bounding_box = postprocess_obj.bounding_box

    if field.lower() in ["total", "total_field", "total_pressure"]:
        pressure_field = _copy.deepcopy(postprocess_obj.total_field_imshow)
    elif field.lower() in ["scattered", "scattered_field", "scattered_pressure"]:
        pressure_field = _copy.deepcopy(postprocess_obj.scattered_field_imshow)
    elif field.lower() in ["incident", "incident_field", "incident_pressure"]:
        pressure_field = _copy.deepcopy(postprocess_obj.incident_field_imshow)
    else:
        raise ValueError(
            "Undefined pressure field, options are total, scattered, and incident."
        )

    if file_name is None:
        file_name_real = None
        file_name_abs = None
    else:
        file_name_real = "real_" + file_name
        file_name_abs = "abs_" + file_name

    scaling_factor, pressure_unit = _convert_pressure_unit(unit)
    pressure_field *= scaling_factor

    if clim is None:
        max_real_pressure = _np.nanmax(_np.abs(_np.real(pressure_field)))
        colormap_lims = (-max_real_pressure, max_real_pressure)
    else:
        if len(clim) != 2:
            raise ValueError("The variable clim must be a tuple of size 2.")
        colormap_lims = clim

    colormap = "seismic"
    axes_labels = _set_pressure_plane(plane_axes)
    if hasattr(postprocess_obj, "domains_edges") and display_edges:
        domains_edges = postprocess_obj.domains_edges
    else:
        domains_edges = None

    fig_p_real = surface_plot(
        _np.real(pressure_field),
        bounding_box,
        axes_labels,
        colormap,
        colormap_lims,
        colorbar_unit="$p_{real}$ [" + pressure_unit + "]",
        domains_edges=domains_edges,
        file_name=file_name_real,
    )

    if clim is None:
        colormap_lims = (0, _np.nanmax(_np.abs(pressure_field)))
    else:
        if len(clim) != 2:
            raise ValueError("The variable clim must be a tuple of size 2.")
        colormap_lims = (0, clim[1])

    colormap = "viridis"

    fig_p_tot = surface_plot(
        _np.abs(pressure_field),
        bounding_box,
        axes_labels,
        colormap,
        colormap_lims,
        colorbar_unit="$p_{abs}$ [" + pressure_unit + "]",
        domains_edges=domains_edges,
        file_name=file_name_abs,
    )

    return fig_p_real, fig_p_tot


def plot_regions(postprocess_obj, display_edges=False, file_name=None):
    """
    Plot the regions in the geometry.

    Parameters
    ----------
    postprocess_obj : optimus.postprocess.method.VisualisePlane
        Optimus postprocess object that includes the visualisation pressure fields
        on a rectangular grid in a plane.
    display_edges : boolean
        Display domains edges, i.e. domains and plane interfaces.
    file_name : str, None
        The file name to export the figure, if specified.

    Returns
    -------
    fig_regions : matplotlib.Figure
        Image show of the regions.
    """

    from matplotlib import cm
    from .common import array_to_imshow

    n_regions = len(postprocess_obj.field.region_legend)
    quantity = array_to_imshow(
        postprocess_obj.field.regions.reshape(postprocess_obj.resolution)
    )

    if hasattr(postprocess_obj, "domains_edges") and display_edges:
        domains_edges = postprocess_obj.domains_edges
    else:
        domains_edges = None

    fig_regions = surface_plot(
        quantity=quantity,
        interpolation="none",
        axes_lims=postprocess_obj.bounding_box,
        axes_labels=_set_pressure_plane(postprocess_obj.plane_axes),
        colormap=cm.get_cmap("tab10", n_regions),
        colormap_lims=(-0.5, n_regions - 0.5),
        colorbar_label=None,
        colorbar_ticks=list(range(n_regions)),
        colorbar_ticklabels=postprocess_obj.field.region_legend,
        domains_edges=domains_edges,
        file_name=file_name,
    )

    return fig_regions


def surface_plot(
    quantity,
    axes_lims,
    axes_labels,
    colormap,
    colormap_lims,
    colorbar_unit,
    domains_edges=None,
    file_name=None,
):
    """2D surface plotting of a mesh grid quantity.

    Parameters
    ----------
    quantity : np.ndarray
        the quantity to be plotted
    axes_lims : list float, tuple float
        List of minima and maxima of h-axis and v-axis of the 2D contour plot,
        in the form of (h_axis_min, h_axis_max, v_axis_min, v_axis_max).
    axes_labels : list str, tuple str
        the labels for h-axis and v-axis of the plot.
    colormap : string
        the name of colormap. all the matplotlib colormaps are supported.
    colormap_lims : list float, tuple float
        The limit values of the colormap.
    colorbar_unit : string
        the label for colorbar
    domains_edges : None, list numpy.ndarray
        If the intersection points of the domains and the visualisation plane is
        passed as a list, they are overlaid to the field plot.
    file_name : None, str
        The file name to export the figure, if specified.

    Returns
    -------
    fig : matplotlib.Figure
        A Matplotlib figure.
    """

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib import pylab as plt
    import numpy as _np

    no_cbarticks = 10
    cbar_ticks = _np.linspace(
        colormap_lims[0], colormap_lims[1], no_cbarticks, endpoint=True
    )
    haxis_label, vaxis_label = axes_labels[0], axes_labels[1]

    fig = plt.figure(figsize=(10, 8))
    ax = plt.gca()

    if domains_edges is not None:
        for i, j in domains_edges:
            plt.plot(i, j, color="black", linestyle="-", linewidth=2)

    ax_image = ax.imshow(
        quantity,
        cmap=colormap,
        clim=colormap_lims,
        extent=axes_lims,
        interpolation="bilinear",
    )

    ax.set_xlabel(haxis_label, size=18)
    ax.set_ylabel(vaxis_label, size=18)
    ax.tick_params(axis="both", which="major", labelsize=14)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cbar = plt.colorbar(ax_image, ticks=cbar_ticks, cax=cax)
    cbar.set_ticklabels(["{:1.1e}".format(i) for i in cbar_ticks])
    cbar.set_label(colorbar_unit, size=18)
    cbar.ax.tick_params(labelsize=14)
    fig.tight_layout()

    if file_name is not None:
        plt.savefig(file_name, bbox_inches="tight")

    plt.show()

    return fig


def _set_pressure_plane(plane_axes):
    axis_labels = ["x [m]", "y [m]", "z [m]"]
    return [axis_labels[i] for i in plane_axes]


def _convert_pressure_unit(pressure_unit):
    units_list = ["Pa", "kPa", "MPa", "GPa"]
    index = [i for i, s in enumerate(units_list) if pressure_unit.lower() == s.lower()]
    if len(index) != 1:
        raise ValueError("Pressure unit " + str(pressure_unit) + " not known.")
    return 10 ** (-3 * index[0]), units_list[index[0]]
