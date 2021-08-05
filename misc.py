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
        exp_dict = exp.get_parameters()
        exp_dict = replace_None_string_with_None(exp_dict)
        if exp.get_system_properties()['name'] == name and exp_dict==params:
            return True
    return False

# # same with (params.items() <= exp.get_parameters().items())
# def compare_dicts(params, exp_params):
#     for key in params:
#         if exp_params[key] != params[key]:
#             return False
#     return True

def tag_picking(project, labels = ['owner', 'created', 'running_time'], tag=['finished_successfully']):
    data = project.get_leaderboard()
    data = data.drop(labels=labels, axis=1)
    succ_experiments =  project.get_experiments(tag=tag)
    return succ_experiments

def replace_None_string_with_None(some_dict):
    return { k: (None if v == 'None' else v) for k, v in some_dict.items() }

def process_dict(some_dict):
    for k, v in some_dict.items():
        if v == 'None':
            some_dict[k] = None

    some_dict['dim']            = int(some_dict['dim'])
    some_dict['rank']           = int(some_dict['rank'])
    some_dict['method_steps']   = int(some_dict['dim'])
    some_dict['seed']           = int(some_dict['seed'])
    some_dict['rho']            = float(some_dict['rho'])
    some_dict['noise']          = float(some_dict['noise'])
    return some_dict

