import shelve, copy

def export_to_file(global_params,
                   transducer_params,
                   model,
                   post_process,
                   file_name = 'optimus_exported_data',file_format = 'db'):
    """ This function exports the simulation results into a file.
    file_name: (string), the export path and the file name in one string.
    file_format: (string), 
    'mat': to save ONLY the post_process results into a MATLAB file.
    'database': (default) to save all the attributes of global parameters, transducer parameters, post processor and pickable objects of the model.
    """
    # deleting attributes that cannot be pickled
    model_copy = copy.deepcopy(model)
    delattr(model_copy,'lhs_model')
    delattr(model_copy,'lhs_prec')
    delattr(model_copy.linear_solver,'_lhs')
    delattr(model_copy,'_source')
    delattr(model_copy,'_system')
    delattr(model_copy,'operator_blocks')
    delattr(model_copy,'prec_operator')
    delattr(model_copy,'sol_gridfunctions')
    delattr(model_copy,'prec_matrix')

    if file_format.lower() == 'mat':
        import scipy.io as sio
        sio.savemat(file_name + '.mat', 
        {'total_field': post_process.total_field,
        'scattered_field': post_process.scattered_field,
        'incident_field': post_process.incident_field,
        'L2_norm_ptot_MPa': post_process.l2_norm_total_field_mpa,
        'gmres_time': model_copy.linear_solver.total_time,
        'gmres_residuals': model.linear_solver.residual_error,
        'gmres_tol': model.linear_solver.tol,
        'gmres_iter_count': model.linear_solver.iter_count, 
        'domain': post_process.bounding_box,
        'visualisation_time': post_process.visualisation_time,
        'resolution_in_points': post_process.resolution,
        'resolution_in__diagonal_ppi':round(post_process._ppi_calculator()),
        '2D_plane_axes': post_process.plane_axes,
        'bounding_box': post_process.bounding_box,
        'plane_offset': post_process.plane_offset,
        'points': post_process.points})
        print('The data are written to the file:',file_name+file_format)
    else:
        db_handle = shelve.open(file_name)
        db_handle['global_parameters'] = global_params
        db_handle['transducer_parameters'] =  transducer_params
        db_handle['model'] = model_copy
        db_handle['post_process'] = post_process
        print('The data are written to the file:',file_name+'.'+file_format)
        print('The list of keys are:', list(db_handle.keys()))
        db_handle.close()
        

def import_from_file(file_name):
    """ This function imports the saved data from a file.
    file_name: (string), this string includes the file name (with path) and the file extension in one string.
    the supported extensions (file formats) are 'mat' and 'db'. 
    """

    import os
    import scipy.io as sio
    import shelve

    file_format = os.path.splitext(file_name)[1]
    if not file_format.lower() in ['.mat','.db']:
        raise TypeError('The file format is unknown, pass a DB or MAT file to import.')
    elif file_format.lower() == '.mat':
        imported_data = sio.loadmat(file_name)
    else:
        db_handle = shelve.open(os.path.splitext(file_name)[0])
        imported_data = dict()
        imported_data = {key:db_handle[key] for key in list(db_handle.keys()) }
        db_handle.close()

    return imported_data




