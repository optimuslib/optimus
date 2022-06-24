# import numpy as np
# import scipy.io as sio
# import time as _time

# def exterior_interior_points_eval(grid, xyz_field, solid_angle_tolerance=0.0):

#     n_field = np.size(xyz_field, axis=1)    
#     elements = grid.leaf_view.elements    
#     vertices = grid.leaf_view.vertices    
#     number_of_elements = grid.leaf_view.entity_count(0)      
#     elem = list(grid.leaf_view.entity_iterator(0))

#     # Prealocate list of element properties
#     element_property = np.zeros(number_of_elements, dtype=np.int)

#     # creates array of element topologies together with property numbers and
#     # creates a list of property numbers
#     element_groups = np.zeros(shape=(4, number_of_elements), dtype=np.int)
#     element_groups[1:4, :] = elements

#     # Retrieves elements from different groups
#     for i in range(number_of_elements):
#         property_number = elem[i].domain
#         element_property[i] = property_number
#         element_groups[0, i] = property_number        

#     # Produce list of element properties                       
#     element_properties = np.array(list(set(element_property)), dtype=np.int)
#     print('Element groups are:')
#     print(element_properties)
       
#     xyz_int = []
#     xyz_ext = []
#     xyz_boundary = []
#     n_int = []
#     n_ext = np.full(xyz_field.shape[1], True, dtype=bool)
#     n_boundary = []
    
#     for i in range(element_properties.size):#-element_properties[0]:
        
#         # Retrieves the block of elements corresponding to the property linked to the index i                
#         elements_trunc = elements[:, element_groups[0, :] == element_properties[i]]
#         # print(elements_trunc)
#         num_elem = elements_trunc.shape[1]
        
#         # Preallocate the arrays for the grid vertices
#         xmesh = np.zeros(shape=(3, num_elem), dtype=float)
#         ymesh = np.zeros(shape=(3, num_elem), dtype=float)
#         zmesh = np.zeros(shape=(3, num_elem), dtype=float)
#         # Populate grid vertices matrices              
#         for k in range(3):         
#             xmesh[k, :] = vertices[0, elements_trunc[k, :]]
#             ymesh[k, :] = vertices[1, elements_trunc[k, :]]
#             zmesh[k, :] = vertices[2, elements_trunc[k, :]]
#         # Obtain coordinates of triangular patch centroids through barycentric method        
#         xcen = np.mean(xmesh,axis=0)
#         ycen = np.mean(ymesh,axis=0)
#         zcen = np.mean(zmesh,axis=0)                  
#         # Preallocate matrix of vectors for triangular patches                            
#         u = np.zeros(shape=(3, num_elem), dtype=float)
#         v = np.zeros(shape=(3, num_elem), dtype=float)
#         # Compute matrix of vectors defining each triangular patch
#         u = np.array([xmesh[1, :] - xmesh[0, :], ymesh[1, :] - ymesh[0, :], zmesh[1, :] - zmesh[0, :]])
#         v = np.array([xmesh[2, :] - xmesh[0, :], ymesh[2, :] - ymesh[0, :], zmesh[2, :] - zmesh[0, :]])
#         # Obtained cross product of u and v vectors
#         u_cross_v = np.cross(u, v, axisa=0, axisb=0, axisc=0)
#         # Obtain magnitude of above cross product
#         u_cross_v_norm = np.linalg.norm(u_cross_v, axis=0)
#         # Obtain outward pointing unit normal vectors for each patch
#         normals = np.divide(u_cross_v, u_cross_v_norm)
#         # Obtain surface area of each patch
#         dS = 0.5*u_cross_v_norm
#         # Preallocate solid angle array
#         Omega = np.zeros(n_field)
#         # Preallocate distances betweem field location and mesh location
#         r = np.zeros(shape=(3, n_field), dtype=float)
#         # Compute solid angle at each field location
#         t0 = _time.time()
#         for j in range(n_field):                        
#             r = np.array([xcen - xyz_field[0, j], ycen - xyz_field[1, j], zcen - xyz_field[2, j]])            
#             r_norm = np.linalg.norm(r, axis=0)
#             r_unit = np.divide(r, r_norm)
#             r_unit_dot_n = np.sum(r_unit*normals, axis=0)
#             Omega[j] = np.sum(r_unit_dot_n * dS / r_norm ** 2) / (4 * np.pi)
#         t1 = _time.time() - t0
#         print('Time to complete (solid angle field parallelisation): ', t1)
         
#         # print(Omega[Omega > 1])
#         # print(Omega[Omega < - solid_angle_tolerance])
#         # Depending on value of solid angle, determine whether field points lie
#         # inside or outside the volume associated with mesh property
#         n_int_tmp = Omega > 0.5 + solid_angle_tolerance
#         # n_ext_tmp = Omega < 0.5 - solid_angle_tolerance
#         n_boundary_tmp = (Omega > 0.5 - solid_angle_tolerance ) & (Omega < 0.5 + solid_angle_tolerance)
        
#         xyz_int.append(xyz_field[:, n_int_tmp])    
#         xyz_boundary.append(xyz_field[:, n_boundary_tmp])        
#         n_int.append(n_int_tmp)         
#         n_boundary.append(n_boundary_tmp)
#         n_ext = n_ext & ((n_int_tmp == False) & (n_boundary_tmp == False))        
    
#     xyz_ext = xyz_field[:, n_ext]
           
#     #sio.savemat('xyz_solid_angle_test',{'xmesh':xmesh, 'ymesh':ymesh, 'zmesh':zmesh,'xyz_int1':xyz_int[0], 'xyz_int2':xyz_int[1], 'xyz_int3':xyz_int[2], 'xyz_int4':xyz_int[3], 'xyz_ext':xyz_ext, 'xcen':xcen, 'ycen':ycen, 'zcen':zcen } )
#     #sio.savemat('xyz_solid_angle2_test.mat',{'xmesh':xmesh, 'ymesh':ymesh, 'zmesh':zmesh,'xyz_int':xyz_int[0], 'xyz_ext':xyz_ext, 'xyz_boundary':xyz_boundary, 'xcen':xcen, 'ycen':ycen, 'zcen':zcen } )
                          
#     return xyz_int, xyz_ext, xyz_boundary, n_int, n_ext, n_boundary

# ##################################################################################################################
import numpy as np
import scipy.io as sio
from functools import partial
import time as _time
import multiprocessing as mp


def Omega_eval(xcen, ycen, zcen, xyz_field, normals, dS, jj):

    r = np.array([xcen - xyz_field[0, jj], ycen - xyz_field[1, jj], zcen - xyz_field[2, jj]])
    r_norm = np.linalg.norm(r, axis=0)
    r_unit = np.divide(r, r_norm)
    r_unit_dot_n = np.sum(r_unit * normals, axis=0)    
    Omega = np.sum(r_unit_dot_n * dS / r_norm**2) / (4 * np.pi)
    #import pdb; pdb.set_trace()
    return Omega

def exterior_interior_points_eval(grid, xyz_field, solid_angle_tolerance=None):
        
    elements = grid.leaf_view.elements
    vertices = grid.leaf_view.vertices
    number_of_elements = grid.leaf_view.entity_count(0)
    elem = list(grid.leaf_view.entity_iterator(0))

    # Prealocate list of element properties
    element_property = np.zeros(number_of_elements, dtype=np.int)

    # creates array of element topologies together with property numbers and
    # creates a list of property numbers
    element_groups = np.zeros(shape=(4, number_of_elements), dtype=np.int)
    element_groups[1:4, :] = elements

    # Retrieves elements from different groups
    for i in range(number_of_elements):
        property_number = elem[i].domain
        element_property[i] = property_number
        element_groups[0, i] = property_number

    # Produce list of element properties
    element_properties = np.array(list(set(element_property)), dtype=np.int)
    print('Element groups are:')
    print(element_properties)

    xyz_int = []
    xyz_ext = []
    xyz_boundary = []
    n_int = []
    n_ext = np.full(xyz_field.shape[1], True, dtype=bool)
    n_boundary = []

    for i in range(element_properties.size):  #-element_properties[0]:

        # Retrieves the block of elements corresponding to the property linked to the index i
        elements_trunc = elements[:, element_groups[0, :] == element_properties[i]]
        # print(elements_trunc)
        num_elem = elements_trunc.shape[1]

        # Preallocate the arrays for the grid vertices
        xmesh = np.zeros(shape=(3, num_elem), dtype=float)
        ymesh = np.zeros(shape=(3, num_elem), dtype=float)
        zmesh = np.zeros(shape=(3, num_elem), dtype=float)
        # Populate grid vertices matrices
        for k in range(3):
            xmesh[k, :] = vertices[0, elements_trunc[k, :]]
            ymesh[k, :] = vertices[1, elements_trunc[k, :]]
            zmesh[k, :] = vertices[2, elements_trunc[k, :]]
        # Obtain coordinates of triangular patch centroids through barycentric method
        xcen = np.mean(xmesh, axis=0)
        ycen = np.mean(ymesh, axis=0)
        zcen = np.mean(zmesh, axis=0)                
                
        # Preallocate matrix of vectors for triangular patches
        u = np.zeros(shape=(3, num_elem), dtype=float)
        v = np.zeros(shape=(3, num_elem), dtype=float)
        # Compute matrix of vectors defining each triangular patch
        u = np.array([xmesh[1, :] - xmesh[0, :], ymesh[1, :] - ymesh[0, :], zmesh[1, :] - zmesh[0, :]])
        v = np.array([xmesh[2, :] - xmesh[0, :], ymesh[2, :] - ymesh[0, :], zmesh[2, :] - zmesh[0, :]])
        # Obtained cross product of u and v vectors
        u_cross_v = np.cross(u, v, axisa=0, axisb=0, axisc=0)
        # Obtain magnitude of above cross product
        u_cross_v_norm = np.linalg.norm(u_cross_v, axis=0)
        # Obtain outward pointing unit normal vectors for each patch
        normals = np.divide(u_cross_v, u_cross_v_norm)
        # Obtain surface area of each patch
        dS = 0.5 * u_cross_v_norm
        
        i_val = np.arange(0, xyz_field.shape[1])
#         print('\n', 70 * '*')
#         print('\n Solid angle field point location parallelisation')
#         #print('\n Number of field chunks: ', N_field_chunks)
#         print('\n', 70 * '*')

        t0 = _time.time()
        
        N_workers = mp.cpu_count()
        func = partial(Omega_eval, xcen, ycen, zcen, xyz_field, normals, dS)
        pool = mp.Pool(N_workers)
        result = pool.starmap(func, zip(i_val))
        pool.close()        
        t1 = _time.time() - t0
        print('Time to complete solid angle field parallelisation: ', t1)
        #import pdb; pdb.set_trace()
        Omega = np.hstack(result)
        if solid_angle_tolerance:
            n_int_tmp = Omega > 0.5 + solid_angle_tolerance        
            n_boundary_tmp = (Omega > 0.5 - solid_angle_tolerance ) & (Omega < 0.5 + solid_angle_tolerance)                               
            xyz_boundary.append(xyz_field[:, n_boundary_tmp])                
            n_boundary.append(n_boundary_tmp)
            n_ext = n_ext & ((n_int_tmp == False) & (n_boundary_tmp == False))
        else:
            n_int_tmp = Omega > 0.5            
            n_ext = n_ext & (n_int_tmp == False)
        
        xyz_int.append(xyz_field[:, n_int_tmp])
        n_int.append(n_int_tmp)


    xyz_ext = xyz_field[:, n_ext]

    return xyz_int, xyz_ext, xyz_boundary, n_int, n_ext, n_boundary



