from . import warper, RAFT

def get_flow_model(model_name, small=True):
    # select base model
    model_name = model_name.lower()
    if model_name == 'raft':
        raft_args ={'small': small,
                    'lr': 1e-5,
                    'mixed_precision': True,
                    'dropout': 0.0,
                    'corr_levels': 4,
                    'corr_radius': 4}
        model = RAFT(raft_args)
    else:
        model = None

    return model
