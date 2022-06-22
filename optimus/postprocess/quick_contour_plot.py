import numpy as np
from matplotlib import pylab as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable


def quick_contour_plot(xmin, xmax, nx, ymin, ymax, ny, pressure_magnitude, pressure_plane=None, pressure_unit=None):
    """
    Quick 2D contour plotting
    """
    initialisedclass = _QuickContourPlot(xmin,xmax,nx,ymin,ymax,ny,pressure_magnitude, \
     pressure_plane,pressure_unit)
    initialisedclass._main()
    return initialisedclass


class _QuickContourPlot(object):
    def __init__(self, xmin, xmax, nx, ymin, ymax, ny, pressure_magnitude, pressure_plane=None, pressure_unit=None):

        self._xmin = xmin
        self._xmax = xmax
        self._nx = nx
        self._ymin = ymin
        self._ymax = ymax
        self._ny = ny
        self._pressure_magnitude = pressure_magnitude
        self._pressure_plane = pressure_plane
        self._pressure_unit = pressure_unit

    def _pressure_unit_check(self):
        if self._pressure_unit in ['mPa', 'Pa', 'kPa', 'MPa', 'GPa']:
            if self._pressure_unit == 'mPa':
                self._scale_factor = 1.E3
            if self._pressure_unit == 'Pa':
                self._scale_factor = 1.
            if self._pressure_unit == 'kPa':
                self._scale_factor = 1.E-3
            if self._pressure_unit == 'MPa':
                self._scale_factor = 1.E-6
            if self._pressure_unit == 'GPa':
                self._scale_factor = 1.E-9
        elif self._pressure_unit is None:
            self._pressure_unit = 'MPa'
            self._scale_factor = 1.E-6

    def _set_pressure_plane(self):
        if self._pressure_plane == 'xy' or self._pressure_plane is None:
            self._axis_labels = ['x [m]', 'y [m]', 'z [m]']
        elif self._pressure_plane == 'yx':
            self._axis_labels = ['y [m]', 'x [m]', 'z [m]']
        elif self._pressure_plane == 'yz':
            self._axis_labels = ['y [m]', 'z [m]', 'x [m]']
        elif self._pressure_plane == 'zy':
            self._axis_labels = ['z [m]', 'y [m]', 'x [m]']
        elif self._pressure_plane == 'zx':
            self._axis_labels = ['z [m]', 'x [m]', 'y [m]']
        elif self._pressure_plane == 'xz':
            self._axis_labels = ['x [m]', 'z [m]', 'y [m]']

    def _plot_2D(self):
        colormap_min = 0
        colormap_max = np.max(self._pressure_magnitude * self._scale_factor)
        no_cbarticks = 6
        colormap = cm.seismic
        colormap1 = cm.jet
        cbar_ticks = np.linspace(colormap_min, colormap_max, no_cbarticks, endpoint=True)
        cbar_ticks_seismic = np.linspace(-colormap_max, colormap_max, no_cbarticks, endpoint=True)
        haxis_label, vaxis_label = self._axis_labels[0], self._axis_labels[1]
        p = np.reshape(self._pressure_magnitude, (self._nx, self._ny)) * self._scale_factor
        fig = plt.figure(figsize=(10, 8))
        ax = plt.gca()
        ax_image = ax.imshow(p,cmap=colormap1,vmin=colormap_min,vmax=colormap_max,extent=[self._xmin, \
      self._xmax, self._ymin, self._ymax],origin='lower')
        ax.set_xlabel(haxis_label, size=18)
        ax.set_ylabel(vaxis_label, size=18)
        ax.tick_params(axis='both', which='major', labelsize=14)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.2)
        cbar = plt.colorbar(ax_image, ticks=cbar_ticks, cax=cax)
        cbar.set_ticklabels(["{:1.1f}".format(i) for i in cbar_ticks])
        cbar.set_label('p [' + self._pressure_unit + ']', size=18)
        cbar.ax.tick_params(labelsize=14)
        fig.tight_layout()
        plt.show()

    def _main(self):
        self._pressure_unit_check()
        self._set_pressure_plane()
        self._plot_2D()
