import numpy as np

class DefaultPostProcessParameters(object):

    VERBOSE = False
    RESOLUTION = [141, 141]
    PLANE_AXES = [0, 1]
    PLANE_OFFSET = 0.0
    BOUNDING_BOX = []
    POINTS = np.array([])
    INDEX_EXTERIOR = None
    INDEX_INTERIOR = None
    OBJ_CHANGED = False
    REPEATED_RUN_FLAG = np.array([])

    PLANE_GRID_X_AXIS_LIMS=[-1, 1],
    PLANE_GRID_Y_AXIS_LIMS=[-1, 1],
    PLANE_GRID_ROTATION_AXIS=[0, 0, 0],
    PLANE_GRID_ROTATION_ANGLE="Pi/2",
    PLANE_GRID_H=0.1

    CREATE_GRID_MODE='2D'

    BEMPP_HMAT_EPS = 1.0e-8
    BEMPP_HMAT_MAX_RANK = 10000
    BEMPP_HMAT_MAX_BLOCK_SIZE = 10000

    PTOT_ALL_FILENAME = './OptimUS_plot3D_ptot_all.msh'
    PTOT_ABS_ALL_FILENAME = './OptimUS_plot3D_ptot_abs_all.msh'
