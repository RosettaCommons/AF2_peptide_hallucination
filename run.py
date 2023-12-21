import os
import sys
import numpy as np
from submodules.oligomer_hallucination.oligomer_hallucination import Protomers, Oligomer
from submodules.oligomer_hallucination.oligomer_hallucination import AA_FREQ
from submodules.oligomer_hallucination.modules.af2_net import setup_models, predict_structure
from submodules.oligomer_hallucination.modules.mutations import mutate

from util.util import select_positions
from util import util
from util.loss import compute_loss
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
import copy

class BoundComplex(Protomers, Oligomer):
    '''
    Class for keeping track of binder sequence and complex predictions
    during binder hallucination.
    '''

    def __init__(self, target_sequence: str, name, length=70, aa_freq={}, binder_sequence=None):
        """
        target_sequence: amino acid sequence of target peptide (to bind)
        length: length of binder peptide
        binder_sequence: Optional, starting amino acid sequence of the binder
        aa_freq: dictonary containing the frequencies of each aa
        """

        self.target_seq = target_sequence.upper()
        assert len(self.target_seq) > 0, "Target sequence must be provided"
        self.length = int(length)
        self.aa_freq = aa_freq

        # Get initial binder sequence
        if binder_sequence:
            assert self.length > 0, "Binder length must be greater than 0"
            self.init_binder_seq = binder_sequence.upper()
        else:
            self.init_binder_seq = ''.join(np.random.choice(list(aa_freq.keys()), size = length, p=list(aa_freq.values())))
        self.binder_length = len(self.init_binder_seq)
        self.target_length = len(self.target_seq)
        self.chain_Ls = [self.binder_length, self.target_length]
        self.init_bound_seq = self.init_binder_seq + self.target_seq
        self.bound_length = len(self.init_bound_seq)

        # Initialize current and try sequences, 
        self.current_binder_seq = self.init_binder_seq
        self.try_binder_seq = self.init_binder_seq

        self.current_bound_seq = self.init_bound_seq
        self.try_seq = self.init_bound_seq

        self.name=name

    def init_scores(self, scores):
        '''Initalise scores'''
        self.init_scores = scores
        self.current_scores = scores
        self.try_scores = scores

    def update_scores(self):
        '''Update current scores to try scores. '''
        self.current_scores = self.try_scores
        
    def assign_scores(self, scores):
        '''Assign try scores. '''
        self.try_scores = scores
    
    def update_scores(self):
        '''Update current scores to try scores.'''
        self.current_scores = copy.deepcopy(self.try_scores)

@hydra.main(version_base=None, config_path='config', config_name='base')
def main(conf: HydraConfig) -> None:
    """
    Main function for running peptide binder hallucination.
    """

    input_conf=conf.input
    output_conf=conf.output
    loss_conf=conf.loss
    model_conf=conf.model
    hallucination_conf=conf.hallucination
    
    os.makedirs(output_conf.out_dir, exist_ok=True)
    if output_conf.cautious and os.path.exists(f'{output_conf.out_dir}/{output_conf.out_prefix}_step_00000.pdb'):
        sys.exit(f'Specified output already exists. Exiting. To overwrite, provide output.cautious=False')

    AA_freq=util.get_aa_freq(AA_FREQ, hallucination_conf.exclude_AA)
    # Initialize BoundComplex object
    boundcomplex = BoundComplex(target_sequence=input_conf.target_sequence, name=conf.output.out_prefix, length=input_conf.binder_length, aa_freq=AA_freq, binder_sequence=input_conf.binder_sequence)

    # Setup AlphaFold2 models.
    model_runners= setup_models(['complex'], model_id=model_conf.model, recycles=model_conf.recycles)

    # Initialize MCMC
    M, current_loss, current_scores = util.initialize_MCMC(conf)

    # Initialize output file
    util.initialize_score_file(conf)

    # Run the hallucination trajectory
    for i in range(hallucination_conf.steps):
        # Update a few things.
        T = hallucination_conf.T_init * (np.exp(np.log(0.5) / hallucination_conf.half_life) ** i) # update temperature
        n_mutations = round(M[i]) # update mutation rate
        if i == 0:
            # Do initial prediction without mutations
            print(f"{'-'*100}")
            print('Starting...')
            
            af2_prediction= predict_structure(boundcomplex,
                                                single_chain=False,
                                                model_runner=model_runners['complex'],
                                                random_seed=0)
            boundcomplex.init_prediction(af2_prediction)
            try_loss, try_scores = compute_loss(loss_conf, boundcomplex)
            boundcomplex.init_loss(try_loss)
            boundcomplex.init_scores(try_scores)

        else:
            # Mutate binder sequences and generate updated bound sequences 
            boundcomplex.assign_mutable_positions(select_positions(n_mutations,
                                                                boundcomplex,
                                                                hallucination_conf.select_positions,
                                                                None))
            boundcomplex.current_sequences={'binder':copy.deepcopy(boundcomplex.current_seq)}
            boundcomplex.assign_mutations(mutate(hallucination_conf.mutation_method,
                                                boundcomplex,
                                                AA_freq))
            boundcomplex.try_seq=boundcomplex.try_sequences['binder']
            boundcomplex.assign_prediction(predict_structure(boundcomplex,
                                        single_chain=False,
                                        model_runner=model_runners['complex'],
                                        random_seed=0))

            try_loss, try_scores = compute_loss(loss_conf, boundcomplex) # calculate the loss
            boundcomplex.assign_loss(try_loss) # assign the loss to the object (for tracking)
            boundcomplex.assign_scores(try_scores) # assign the scores to the object (for tracking)
        
        # Evaluation step
        accepted=util.accept_or_reject(boundcomplex, T, i)
        if accepted:
            util.write_outputs(boundcomplex, output_conf, i)

        # Save scores
        util.append_score_file(i, accepted, T, n_mutations, try_loss, try_scores, conf)

if __name__ == "__main__":
    main()
    print("Done!")
