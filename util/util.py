import numpy as np
import copy
import sys
from alphafold.common import protein

def select_positions(n_mutations, boundcomplex, select_positions, select_position_params):
    '''
    Select mutable positions in the binder based on a specific method.
    Returns a dictionary of binder with associated array indicating mutable positions.
    '''

    mutable_positions = {}

    if select_positions == 'random':
        # Choose positions randomly.
        mutable_positions['binder'] = np.random.choice(range(len(boundcomplex.current_binder_seq)), size=n_mutations, replace=False)

    elif select_positions == 'plddt':
        # Choose positions based on lowest plddt in binder sequence.
        # First/last three positions of binder are choice frequency adjusted to avoid picking N/C term every time (they tend to score much lower).

        mutate_plddt_quantile = 0.5 # default worst pLDDT quantile to mutate.
    
        # Get plddts from sequence object (binder)  
        plddts = boundcomplex.current_prediction_results["plddt"]
    
        # Take just binder segment
        plddts = plddts[:boundcomplex.binder_length,]
    
        # Weights associated with each position in the binder.
        # to account for termini systematically scoring worse in pLDDT.
        weights = np.array([0.25, 0.5, 0.75] + [1] * (boundcomplex.binder_length - 6) + [0.75, 0.5, 0.25])

        n_potential = round(boundcomplex.binder_length * mutate_plddt_quantile)
        potential_sites = np.argsort(plddts)[:n_potential]

        # Select mutable sites
        sub_w = weights[potential_sites]
        sub_w = [w/np.sum(sub_w) for w in sub_w]
        sites = np.random.choice(potential_sites, size=n_mutations, replace=False, p=sub_w)

        mutable_positions['binder'] = sites

    return mutable_positions

def get_aa_freq(AA_freq: dict, exclude_AA: str):
    """
    Gets aa frequences from a dictionary, and excludes any user-specified amino acids
    """
    for aa in exclude_AA:
        del AA_freq[aa]

    # Re-compute frequencies to sum to 1.
    sum_freq = np.sum(list(AA_freq.values()))
    adj_freq = [f/sum_freq for f in list(AA_freq.values())]
    return dict(zip(AA_freq, adj_freq))

def initialize_MCMC(conf):
    """
    Initializes things pre MCMC
    Outputs:
        - M: mutation rate at each step
        - current_loss: inf at start
        - current_scores: inf at start
    """
    Mi, Mf = conf.hallucination.mutation_rate.split('-')
    M = np.linspace(int(Mi), int(Mf), conf.hallucination.steps) # stepped linear decay of the mutation rate

    current_loss = np.inf
    current_scores={}
    for loss,weight in conf.loss.items():
        current_scores[loss] = [np.inf, weight]
    
    return M, current_loss, current_scores

def initialize_score_file(conf) -> None:
    """
    Initializes the score file
    """
    with open(f'{conf.output.out_dir}/scores.out', 'w') as f:
        f.write(f'# {conf}\n')
        f.write(f'step accepted temperature mutations total_loss {" ".join(conf.loss.keys())}\n')

def append_score_file(i, accepted, T, n_mutations, try_loss, try_scores, conf) -> None:
    """
    Appends scores to the score file
    """
    with open(f'{conf.output.out_dir}/scores.out','a') as f:
        f.write(f'{i} {accepted} {T} {n_mutations} {try_loss} {" ".join(list(map(str, try_scores.values())))}\n')

def accept_or_reject(boundcomplex, T, step):
    """
    Accepts based on Metropolis criterion
    """
    # If the new solution is better, accept it.
    delta = boundcomplex.try_loss - boundcomplex.current_loss # all losses must be defined such that optimising equates to minimising.
    if delta < 0:
        accepted = True
        print(f'Step {step:05d}: change accepted >> LOSS {boundcomplex.current_loss:2.3f} --> {boundcomplex.try_loss:2.3f}')

    # If the new solution is not better, accept it with a probability of e^(-cost/temp).
    elif np.random.uniform(0, 1) < np.exp( -delta / T):
        accepted = True
        print(f'Step {step:05d}: change accepted despite not improving >> LOSS {boundcomplex.current_loss:2.3f} --> {boundcomplex.try_loss:2.3f}')
    else:
        accepted = False
        print(f'Step {step:05d}: change rejected >> LOSS {boundcomplex.current_loss:2.3f} !-> {boundcomplex.try_loss:2.3f}')

    if accepted:
        boundcomplex.current_seq=copy.deepcopy(boundcomplex.try_seq) # accept sequence changes
        for key, item in boundcomplex.try_scores.items():
            print(f' > {key} {boundcomplex.current_scores.get(key):2.3f} --> {item:2.3f}')

        boundcomplex.update_prediction() # accept score/structure changes
        boundcomplex.update_loss() # accept loss change
        boundcomplex.update_scores() # accept scores change
    print('=' * 70)
    sys.stdout.flush()
    return accepted

def write_outputs(boundcomplex, conf, i) -> None:
    """
    Dumps outputs to the output folder. 
    """
    pdb_lines = relabel_chains(protein.to_pdb(boundcomplex.current_unrelaxed_structure).split('\n'))
    with open(f'{conf.out_dir}/{conf.out_prefix}_step_{i:05d}.pdb', 'w') as f:
        f.write('MODEL     1\n')
        f.write('\n'.join(pdb_lines))
        f.write('\nENDMDL\nEND\n')
    
    if conf.output_pae:
        np.save(f'{conf.out_dir}/{conf.out_prefix}_step_{i:05d}.npy', boundcomplex.current_prediction_results['predicted_aligned_error'])

def relabel_chains(pdb_lines):
    """
    Identify chain breaks and reassign chain letters
    """
    chains = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    # Identify chain breaks and re-assign chains correctly before generating PDB file.
    split_lines = [l.split() for l in pdb_lines if 'ATOM' in l]
    split_lines = np.array([l[:4] + [l[4][0]] + [l[4][1:]] + l[5:] if len(l)<12 else l for l in split_lines]) # chain and resid no longer space-separated at high resid.
    splits = np.argwhere(np.diff(split_lines.T[5].astype(int))>1).flatten() + 1 # identify idx of chain breaks based on resid jump.
    splits = np.append(splits, len(split_lines))
    chains = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    chain_str = ''
    prev = 0
    for ch, resid in enumerate(splits): # make new chain string
        length = resid - prev
        chain_str += chains[ch] * length
        prev = resid
    atom_lines = [l for l in pdb_lines if 'ATOM' in l]
    # generate chain-corrected PDB lines.
    new_lines = [f'{l[:21]}{chain_str[k]}{l[22:]}' for k, l in enumerate(atom_lines)] 
    return new_lines
