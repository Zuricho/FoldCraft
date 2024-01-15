# Packages in original python
import enum
import json
import os
import pathlib
import pickle
import random
import shutil
import sys
import time
from typing import Any, Dict, Mapping, Union

# Package needs to install
from absl import app
from absl import flags
from absl import logging
import jax.numpy as jnp
import numpy as np

# Package in this project
from foldcraft.zurtein import protein
from foldcraft.zurtein import residue_constants
from foldcraft.alphafold.model import config
from foldcraft.alphafold.model import data
from foldcraft.alphafold.model import model




def _jnp_to_np(output: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively changes jax arrays to numpy arrays."""
    for k, v in output.items():
        if isinstance(v, dict):
            output[k] = _jnp_to_np(v)
        elif isinstance(v, jnp.ndarray):
            output[k] = np.array(v)
    return output



def alphafold_setup(model_name, 
                    num_ensemble=1, 
                    num_recycle=3,
                    parameter_path="/Volumes/Pacifica/Storage/alphafold_params/AF_2_3_params",
                    prediction_id=1,
                    ):
    """
    model_name: str      
    - model_1, model_2, model_3, model_4, model_5
    - model_1_ptm, model_2_ptm, model_3_ptm, model_4_ptm, model_5_ptm
    - model_1_multimer, model_2_multimer, model_3_multimer, model_4_multimer, model_5_multimer
    """


    # set for logging
    # logging.set_verbosity(logging.INFO)


    model_config = config.model_config(model_name)
        
    model_config.data.eval.num_ensemble = num_ensemble
    model_config.model.num_recycle = num_recycle  # FLAGS.recycling
    model_config.data.common.num_recycle = num_recycle  # FLAGS.recycling
    
    model_params = data.get_model_haiku_params(
            model_name=model_name, parameter_path=parameter_path)   # FLAGS.parameter_path
    model_runner = model.RunModel(model_config, model_params)

    return model_runner



def alphafold_load_feature(
        fasta_name,
        output_dir_base = 'output',
):
    # timings = {}
    output_dir_base = output_dir_base
    fasta_name = fasta_name
    output_dir = os.path.join(output_dir_base, fasta_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get features.
    # We already have feature.pkl file, skip the MSA and template finding step
    t_0 = time.time()
    features_output_path = os.path.join(output_dir, 'features.pkl')
    feature_dict = pickle.load(open(features_output_path, 'rb'))


    # timings['features'] = time.time() - t_0
    print(f"[Time] Load feature: {time.time()-t_0:.2f} s")

    return feature_dict



def alphafold_predict_structure(
        model_runner,
        model_name,
        feature_dict,
        fasta_name,
        random_seed=42,
        output_dir_base = 'output',
):
    # ranking_confidences = {}
    output_dir = os.path.join(output_dir_base, fasta_name)


    # logging.info('Running model %s on %s', model_name, fasta_name)
    t_0 = time.time()
    processed_feature_dict = model_runner.process_features(
        feature_dict, random_seed=random_seed)
    # timings[f'process_features_{model_name}'] = time.time() - t_0
    print(f"[Time] Process feature: {time.time()-t_0:.2f} s")

    t_0 = time.time()
    prediction_result = model_runner.predict(processed_feature_dict,
                                             random_seed=random_seed)
    t_diff = time.time() - t_0
    print(f"[Time] Predict and compile: {t_diff:.2f} s")

    plddt = prediction_result['plddt']
    # ranking_confidences[model_name] = prediction_result['ranking_confidence']

    # Remove jax dependency from results.
    np_prediction_result = _jnp_to_np(dict(prediction_result))
    

    # Add the predicted LDDT in the b-factor column.
    # Note that higher predicted LDDT value means higher model confidence.
    plddt_b_factors = np.repeat(
        plddt[:, None], residue_constants.atom_type_num, axis=-1)
    unrelaxed_protein = protein.from_prediction(
        features=processed_feature_dict,
        result=prediction_result,
        b_factors=plddt_b_factors,
        remove_leading_feature_dimension=not model_runner.multimer_mode)

    # unrelaxed_proteins[model_name] = unrelaxed_protein
    unrelaxed_pdb = protein.to_pdb(unrelaxed_protein)
    unrelaxed_pdb_path = os.path.join(output_dir, f'{fasta_name}_{model_name}.pdb')
    with open(unrelaxed_pdb_path, 'w') as f:
        f.write(unrelaxed_pdb)


    return np_prediction_result







