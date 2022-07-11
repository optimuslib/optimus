import shelve, copy


def export_to_file(
    model,
    post_process,
    global_parameters,
    file_name="optimus_exported_data",
    file_format="db",
):
    """
    To export the simulation results into a file.

    Parameters
    -------------
    file_name : str
        the export path and the file name in one string.
    file_format : str
        'mat': to save ONLY the post_process results into a MATLAB file.
        'db': (default) to save all the attributes of global parameters, source parameters, post processor and pickable objects of the model.
    """
    model_copy = copy.deepcopy(model)
    delattr(model_copy, "continous_operator")
    delattr(model_copy, "discrete_operator")
    delattr(model_copy, "discrete_preconditioner")
    delattr(model_copy, "lhs_discrete_system")

    postprocess_copy = copy.deepcopy(post_process)
    delattr(postprocess_copy, "model")

    if file_format.lower() == "mat":
        import scipy.io as sio

        sio.savemat(
            file_name + ".mat",
            {
                "total_field": post_process.total_field,
                "scattered_field": post_process.scattered_field,
                "incident_field": post_process.incident_field,
                "L2_norm_ptot_MPa": post_process.l2_norm_total_field_mpa,
                "gmres_iter_count": model.iteration_count,
                "points": post_process.points,
            },
        )
        print("The data are written to the file:", file_name + file_format)
    else:
        db_handle = shelve.open(file_name)
        db_handle["global_parameters"] = global_parameters
        db_handle["source"] = model_copy.source
        db_handle["model"] = model_copy
        db_handle["post_process"] = postprocess_copy
        print("The data are written to the file:", file_name + "." + file_format)
        print("The list of keys are:", list(db_handle.keys()))
        db_handle.close()


def import_from_file(file_name):
    """
    To import the saved data from a file.

    Parameters
    ------------
    file_name : str
        this string includes the file name (with path) and the file extension in one string. The supported extensions are 'mat' and 'db'.

    Returns
    ------------
    imported_data : dict
        dictionary of different objects imported from the data file.
    """

    import os
    import scipy.io as sio
    import shelve

    file_format = os.path.splitext(file_name)[1]
    if not file_format.lower() in [".mat", ".db"]:
        raise TypeError("The file format is unknown, pass a DB or MAT file to import.")
    elif file_format.lower() == ".mat":
        imported_data = sio.loadmat(file_name)
    else:
        db_handle = shelve.open(os.path.splitext(file_name)[0])
        imported_data = dict()
        imported_data = {key: db_handle[key] for key in list(db_handle.keys())}
        db_handle.close()

    return imported_data
