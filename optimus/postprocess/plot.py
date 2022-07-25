"""Functionality to plot pressure fields."""


def plot_pressure_field(postprocess_obj, field="total", unit="Pa", clim=()):

    """
    2D contour plotting of pressure fields of an optimus post process object

    Parameters
    ----------
    postprocess_obj : optimus.postprocess.PostProcess
        optimus postprocess object that includes the visualisation pressure fields
    field : string
        The pressure field to be plotted.
        The options are:
            1. total, total_field, or total_pressure;
            2. scattered, scattered_pressure, or scattered_field.
    unit: string
        Pressure unit. the pressure fields are scaled accordingly.
        Options are: Pa, kPa, MPa and GPa.
    clim : tuple
        colorbar limits: (clim_min, clim_max). Must be of the same units as pressure fields.

    Returns
    ----------
    fig_p_real, fig_p_tot
        Contour plots of the real and absolute value of the pressure field.
    """
    import numpy as _np

    plane_axes = postprocess_obj.plane_axes
    bounding_box = postprocess_obj.bounding_box

    if field.lower() in ["total", "total_field", "total_pressure"]:
        pressure_field = postprocess_obj.total_field_imshow
    elif field.lower() in ["scattered", "scattered_field", "scattered_pressure"]:
        pressure_field = postprocess_obj.scattered_field_imshow
    else:
        raise ValueError("Undefined pressure field, options are total and scattered.")

    scaling_factor, pressure_unit = _convert_pressure_unit(unit)
    pressure_field *= scaling_factor

    if clim:
        colormap_lims = clim
    else:
        max_real_pressure = _np.nanmax(_np.abs(_np.real(pressure_field)))
        colormap_lims = (-max_real_pressure, max_real_pressure)
    colormap = "seismic"
    axes_labels = _set_pressure_plane(plane_axes)
    if hasattr(postprocess_obj, "domains_edges"):
        domains_edges = postprocess_obj.domains_edges
    else:
        domains_edges = False

    fig_p_real = contour_plot(
        _np.real(pressure_field),
        bounding_box,
        axes_labels,
        colormap,
        colormap_lims,
        colorbar_unit="$p_{real}$ [" + pressure_unit + "]",
        domains_edges=domains_edges,
    )

    if clim:
        colormap_lims = (0, clim[1])
    else:
        colormap_lims = (0, _np.nanmax(_np.abs(pressure_field)))
    colormap = "viridis"
    fig_p_tot = contour_plot(
        _np.abs(pressure_field),
        bounding_box,
        axes_labels,
        colormap,
        colormap_lims,
        colorbar_unit="$p_{abs}$ [" + pressure_unit + "]",
        domains_edges=domains_edges,
    )

    return fig_p_real, fig_p_tot


def contour_plot(
    quantity,
    axes_lims,
    axes_labels,
    colormap,
    colormap_lims,
    colorbar_unit,
    domains_edges=False,
):
    """
    2D contour plotting of a mesh grid quantity

    Parameters
    ----------
    quantity : np.ndarray
        the quantity to be plotted
    axes_lims : list[float], tuple[float]
        List of minima and maxima of h-axis and v-axis of the 2D contour plot,
        in the form of [h_axis_min, h_axis_max, v_axis_min, v_axis_max].
    axes_labels : list[str], tuple[str]
        the labels for h-axis and v-axis of the plot.
    colormap : string
        the name of colormap. all the matplotlib colormaps are supported.
    colormap_lims : list[float], tuple[float]
        The limit values of the colormap.
    colorbar_unit : string
        the label for colorbar
    domains_edges : bool, list[bool]
        if the intersection points of the domains and the visualisation plane
        is passed as a list, they are overlaid to the field plot.

    Returns
    ----------
    fig : plt.figure
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
    ax_image = ax.imshow(
        quantity,
        cmap=colormap,
        clim=colormap_lims,
        extent=axes_lims,
        interpolation="bilinear",
    )

    if domains_edges:
        if len(domains_edges):
            for i, j in domains_edges:
                plt.plot(i, j, color="black", linestyle="-", linewidth=2)

    ax.set_xlabel(haxis_label, size=18)
    ax.set_ylabel(vaxis_label, size=18)
    ax.tick_params(axis="both", which="major", labelsize=14)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cbar = plt.colorbar(ax_image, ticks=cbar_ticks, cax=cax)
    cbar.set_ticklabels(["{:1.1f}".format(i) for i in cbar_ticks])
    cbar.set_label(colorbar_unit, size=18)
    cbar.ax.tick_params(labelsize=14)
    fig.tight_layout()
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
