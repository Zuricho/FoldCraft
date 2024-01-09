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




def main():
    model_runners = {}
    model_names = (
        'model_1',
#       'model_2',
#       'model_3',
#       'model_4',
#       'model_5',
    )
    for model_name in model_names:
        model_config = config.model_config(model_name)
        
        model_config.data.eval.num_ensemble = 1
        model_config.model.num_recycle = 3  # FLAGS.recycling
        model_config.data.common.num_recycle = 3  # FLAGS.recycling
        
        model_params = data.get_model_haiku_params(
                model_name=model_name, parameter_path="data/params")   # FLAGS.parameter_path
        model_runner = model.RunModel(model_config, model_params)

        num_predictions_per_model = 1
        for i in range(num_predictions_per_model):
            model_runners[f'{model_name}_pred_{i}'] = model_runner


    random_seed = 1
    logging.info('Using random seed %d for the data pipeline', random_seed)


    # predict_structure function
    # logging.info('Predicting %s', fasta_name)
    timings = {}
    output_dir_base = 'output'
    fasta_name = "test"
    output_dir = os.path.join(output_dir_base, fasta_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get features.
    # We already have feature.pkl file, skip the MSA and template finding step
    t_0 = time.time()
    features_output_path = os.path.join(output_dir, 'features.pkl')
    feature_dict = pickle.load(open(features_output_path, 'rb'))


    timings['features'] = time.time() - t_0


    unrelaxed_pdbs = {}
    unrelaxed_proteins = {}
    relaxed_pdbs = {}
    relax_metrics = {}
    ranking_confidences = {}

    # Run the models.
    num_models = len(model_runners)
    for model_index, (model_name, model_runner) in enumerate(model_runners.items()):
        logging.info('Running model %s on %s', model_name, fasta_name)
        t_0 = time.time()
        model_random_seed = model_index + random_seed * num_models   # model random seed
        processed_feature_dict = model_runner.process_features(
            feature_dict, random_seed=model_random_seed)
        timings[f'process_features_{model_name}'] = time.time() - t_0

        t_0 = time.time()
        prediction_result = model_runner.predict(processed_feature_dict,
                                                 random_seed=model_random_seed)
        t_diff = time.time() - t_0
        timings[f'predict_and_compile_{model_name}'] = t_diff    # note: this time contains compile
        logging.info(
            'Total JAX model %s on %s predict time (includes compilation time, see --benchmark): %.1fs',
            model_name, fasta_name, t_diff)


    plddt = prediction_result['plddt']
    ranking_confidences[model_name] = prediction_result['ranking_confidence']

    # Remove jax dependency from results.
    np_prediction_result = _jnp_to_np(dict(prediction_result))

    # Save the model outputs.
    result_output_path = os.path.join(output_dir, f'result_{model_name}.pkl')
    with open(result_output_path, 'wb') as f:
        pickle.dump(np_prediction_result, f, protocol=4)

    # Add the predicted LDDT in the b-factor column.
    # Note that higher predicted LDDT value means higher model confidence.
    plddt_b_factors = np.repeat(
        plddt[:, None], residue_constants.atom_type_num, axis=-1)
    unrelaxed_protein = protein.from_prediction(
        features=processed_feature_dict,
        result=prediction_result,
        b_factors=plddt_b_factors,
        remove_leading_feature_dimension=not model_runner.multimer_mode)

    unrelaxed_proteins[model_name] = unrelaxed_protein
    unrelaxed_pdbs[model_name] = protein.to_pdb(unrelaxed_protein)
    unrelaxed_pdb_path = os.path.join(output_dir, f'unrelaxed_{model_name}.pdb')
    with open(unrelaxed_pdb_path, 'w') as f:
        f.write(unrelaxed_pdbs[model_name])

    # Rank by model confidence.
    ranked_order = [
        model_name for model_name, confidence in
        sorted(ranking_confidences.items(), key=lambda x: x[1], reverse=True)]

if __name__ == '__main__':
    main()

