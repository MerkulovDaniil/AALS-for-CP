# import tensorly as tl

# init='svd'
# svd='numpy_svd'
# orthogonalise = False
# normalize_factors = True
# random_state=False


# weights, factors = initialize_cp(tensor, rank, init=init, svd=svd, 
#                                 random_state=random_state,
#                                 normalize_factors=normalize_factors)

######################################################################


def check_exp(project, name, params):
    succExperiments =  project.get_experiments(tag=['finished_successfully', name])
    for exp in succExperiments:
        if exp.get_system_properties()['name'] == name and exp.get_parameters()==params:
            return True
    return False

def compare_dicts(params, exp_params):
    for key in params:
        if exp_params[key] != params[key]:
            return False
    return True