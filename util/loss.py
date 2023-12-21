import numpy as np
import scipy
from submodules.oligomer_hallucination.modules.losses import get_coord
from collections import OrderedDict

def compute_loss(conf, boundcomplex):
    """
    Computes losses as defined by the config file
    """
    losses=OrderedDict()
    for loss_name in conf:
        loss_function = globals().get(loss_name, None)
        if loss_function is not None and callable(loss_function):
            losses[loss_name] = loss_function(boundcomplex)
        else:
            raise ValueError(f"Loss function {loss_name} not found")
    total_loss=combine_loss(losses, conf)
    return total_loss, losses

def combine_loss(losses, conf):
    """
    Combines losses, such that minimizing each loss makes sense.
    Applies loss weighting as defined in the config
    """
    TO_MAX=['plddt','ptm','contact_prob']
    TO_MIN=['rog','iPAE']
    total_loss=0
    for loss_name,loss in losses.items():
        if loss_name in TO_MAX:
            total_loss += (1-loss) * conf[loss_name]
        elif loss_name in TO_MIN:
            total_loss += loss * conf[loss_name]
        else:
            raise ValueError(f"Loss function {loss_name} must be defined as to maximise or minimise")
    return total_loss

def plddt(boundcomplex):
    """pLDDT from AF2"""
    af2_results=boundcomplex.try_prediction_results
    return np.mean(af2_results["plddt"])

def ptm(boundcomplex):
    """pTM from AF2"""
    af2_results=boundcomplex.try_prediction_results
    return af2_results["ptm"]

def rog(boundcomplex):
    """
    Calculates radius of gyration
    """
    c = np.vstack(get_coord('CA', boundcomplex)[:,2])

    center_of_mass = np.mean(c, axis=0)
    c-=center_of_mass

    distances_squared=np.sum(c**2, axis=1)
    rog = np.sqrt(np.mean(distances_squared))

    # Normalization to theoretical sphere. 10 is arbitrary.
    ideal_rog = np.cbrt((3 * len(distances_squared)) / (4 * np.pi)) * 10
    rog /= ideal_rog

    return rog

def contact_prob(boundcomplex):
    """
    Calculates probability mass under 8A
    """
    af2_results=boundcomplex.try_prediction_results
    probs = scipy.special.softmax(af2_results['distogram']['logits'], axis=-1)
    probs8A = np.sum(probs[:,:,:18], axis=-1)[boundcomplex.binder_length:,:boundcomplex.binder_length]

    # sort in descending order along axis 1
    probs8A = -np.sort(-probs8A, axis=1)
    assert probs8A.shape[1] == boundcomplex.binder_length
    return np.mean(probs8A[:,0])

def iPAE(boundcomplex):
    """
    Calculates pAE over the interface, normalising for initial pAE
    """
    init_results=boundcomplex.init_prediction_results
    try_results=boundcomplex.try_prediction_results
    sub_mat_init = []
    sub_mat_try = []
    sub_mat_init.append(init_results['predicted_aligned_error'][boundcomplex.binder_length:,:boundcomplex.binder_length])
    sub_mat_init.append(init_results['predicted_aligned_error'][:boundcomplex.binder_length,boundcomplex.binder_length:])
    sub_mat_try.append(try_results['predicted_aligned_error'][boundcomplex.binder_length:,:boundcomplex.binder_length])
    sub_mat_try.append(try_results['predicted_aligned_error'][:boundcomplex.binder_length,boundcomplex.binder_length:])
    norm =  np.mean([np.mean(sub_m) for sub_m in sub_mat_init])
    score_sub_pae = np.mean([np.mean(sub_m) for sub_m in sub_mat_try]) / norm
    return score_sub_pae

