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
# list: ml_collections, absl, jax, numpy

# Package in this project
from alphafold.model import config
from alphafold.model import data
from alphafold.model import model


def predict_structure():
    """Predicts structure using AlphaFold for the given sequence."""
    pass


def main():
    model_runners = {}
    model_names = (
        'model_1',
        'model_2',
        'model_3',
        'model_4',
        'model_5',
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
    # logging.info('Using random seed %d for the data pipeline', random_seed)

    # Predict structure for each of the sequences.
    # predict_structure(
    #         fasta_path=fasta_path,
    #         fasta_name=fasta_name,
    #         output_dir_base=FLAGS.output_dir,
    #         data_pipeline=data_pipeline,
    #         model_runners=model_runners,
    #         amber_relaxer=amber_relaxer,
    #         benchmark=FLAGS.benchmark,
    #         random_seed=random_seed,
    #         models_to_relax=FLAGS.models_to_relax,
    #         run_feature = FLAGS.run_feature)
    # logging.info('%s AlphaFold structure prediction COMPLETE', fasta_name)



if __name__ == '__main__':
    main()

