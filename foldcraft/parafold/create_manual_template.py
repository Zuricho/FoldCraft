# create_manual_feature.py can create create empty feature.pkl file for parafold, make the parafold skip the MSA and template part
import numpy as np
import pickle
import residue_constants
import os
import pathlib

from absl import app
from absl import flags


flags.DEFINE_list(
    'fasta_paths', None, 'Paths to FASTA files, each containing a prediction '
    'target that will be folded one after another. If a FASTA file contains '
    'multiple sequences, then it will be folded as a multimer. Paths should be '
    'separated by commas. All FASTA paths must have a unique basename as the '
    'basename is used to name the output directories for each prediction.')
flags.DEFINE_string('output_dir', None, 'Path to a directory that will '
                    'store the results.')
flags.DEFINE_string('template_path', None, 'Path to a PDB file that will '
                    'be used as template.')


FLAGS = flags.FLAGS


def parse_fasta(fasta_string: str):
    """Parses FASTA string and returns list of strings with amino-acid sequences.

    Arguments:
      fasta_string: The string contents of a FASTA file.

    Returns:
      A tuple of two lists:
      * A list of sequences.
      * A list of sequence descriptions taken from the comment lines. In the
        same order as the sequences.
    """
    sequences = []
    descriptions = []
    index = -1
    for line in fasta_string.splitlines():
        line = line.strip()
        if line.startswith('>'):
            index += 1
            descriptions.append(line[1:])  # Remove the '>' at the beginning.
            sequences.append('')
            continue
        elif not line:
            continue  # Skip blank lines.
        sequences[index] += line

    return sequences, descriptions


def make_sequence_features(
    sequence: str, description: str, num_res: int):
    """Constructs a feature dict of sequence features."""
    features = {}
    features['aatype'] = residue_constants.sequence_to_onehot(
          sequence=sequence,
          mapping=residue_constants.restype_order_with_x,
          map_unknown_to_x=True)
    features['between_segment_residues'] = np.zeros((num_res,), dtype=np.int32)
    features['domain_name'] = np.array([description.encode('utf-8')],
                                        dtype=np.object_)
    features['residue_index'] = np.array(range(num_res), dtype=np.int32)
    features['seq_length'] = np.array([num_res] * num_res, dtype=np.int32)
    features['sequence'] = np.array([sequence.encode('utf-8')], dtype=np.object_)
    return features



def make_empty_msa_features(
    sequence: str, num_res: int):
    """Constructs a feature dict of empty MSA features."""
    int_msa = []
    num_alignments = 1
    species_ids = b''

    # Add the query sequence.
    int_msa.append([residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence])


    features = {}
    features['deletion_matrix_int'] = np.zeros((1,num_res), dtype=np.int32)
    features['msa'] = np.array(int_msa, dtype=np.int32)
    features['num_alignments'] = np.array(
        [num_alignments] * num_res, dtype=np.int32)
    features['msa_species_identifiers'] = np.array(species_ids, dtype=np.object_)
    return features


TEMPLATE_FEATURES = {
    'template_aatype': np.float32,
    'template_all_atom_masks': np.float32,
    'template_all_atom_positions': np.float32,
    'template_domain_names': object,
    'template_sequence': object,
    'template_sum_probs': np.float32,
}


def make_empty_template_features(
    sequence: str, num_res: int):
    """Constructs a feature dict of empty template features."""
    
    template_features = {}
    for template_feature_name in TEMPLATE_FEATURES:
        template_features[template_feature_name] = []

    for name in template_features:
        # Make sure the feature has correct dtype even if empty.
        template_features[name] = np.array([], dtype=TEMPLATE_FEATURES[name])

    return template_features



def make_manual_template_features(pdb_path: str):
    """Read PDB file, return a list containing atoms coordinates and positions"""
    with open(pdb_path, 'r') as f:
        lines = f.readlines()
    
    # get pdb id and chain id
    file_name = pdb_path.split('/')[-1].split('.')[0]

    atoms = []
    for line in lines:
        if line.startswith('ATOM'):
            atom = line[12:16].strip()
            res_name = line[17:20].strip()
            res_id = int(line[22:26])
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            atoms.append([atom, res_name, res_id, x, y, z])
    
    # get the number of residues
    num_res = 0
    for atom in atoms:
        if atom[2] > num_res:
            num_res = atom[2]
    

    # create residue list
    residues = [[] for i in range(num_res)]
    for atom in atoms:
        residues[atom[2]-1].append(atom)

    sequence = "".join([residue_constants.restype_3to1[residues[i][0][1]] for i in range(num_res)])
    aa_type = residue_constants.sequence_to_onehot(sequence, residue_constants.HHBLITS_AA_TO_ID)
    

    
    
    all_positions = np.zeros([num_res, residue_constants.atom_type_num, 3])
    all_positions_mask = np.zeros([num_res, residue_constants.atom_type_num], dtype=np.int64)


    for res_index in range(num_res):
        pos = np.zeros([residue_constants.atom_type_num, 3], dtype=np.float32)
        mask = np.zeros([residue_constants.atom_type_num], dtype=np.float32)
        
        for atom in residues[res_index]:
            if atom[0] not in residue_constants.atom_types:
                continue
            atom_type_index = residue_constants.atom_types.index(atom[0])
            pos[atom_type_index, :] = atom[3:]
            mask[atom_type_index] = 1.0
        
        all_positions[res_index, :, :] = pos
        all_positions_mask[res_index, :] = mask


    return (
      {
          'template_all_atom_positions': np.array([all_positions], dtype=TEMPLATE_FEATURES['template_all_atom_positions']),
          'template_all_atom_masks': np.array([all_positions_mask], dtype=TEMPLATE_FEATURES['template_all_atom_masks']),
          'template_sequence': np.array([sequence.encode()], dtype=TEMPLATE_FEATURES['template_sequence']),
          'template_aatype': np.array([aa_type], dtype=TEMPLATE_FEATURES['template_aatype']),
          'template_domain_names': np.array([f'{file_name.lower()}'.encode()], dtype=TEMPLATE_FEATURES['template_domain_names']),
          'template_sum_probs': np.array([[100.0]], dtype=TEMPLATE_FEATURES['template_sum_probs']),
      })



def create_manual_feature(fasta_path, template_path):
    # Load input sequence.
    with open(fasta_path) as f:
        input_fasta_str = f.read()
        input_seqs, input_descs = parse_fasta(input_fasta_str)
        if len(input_seqs) != 1:
          raise ValueError(
              f'More than one input sequence found in {fasta_path}.')
        input_sequence = input_seqs[0]
        input_description = input_descs[0].split()[0]
        num_res = len(input_sequence)

    # Generate feature: sequence part
    feature_dict = make_sequence_features(
                   sequence=input_sequence,
                   description=input_description,
                   num_res=num_res)

    # Generate feature: MSA part
    feature_dict.update(make_empty_msa_features(
                    sequence=input_sequence,
                    num_res=num_res))

    # Generate feature: template part, take one PDB file as template
    feature_dict.update(make_manual_template_features(
                    pdb_path=template_path))
    
    return feature_dict







def manual_feature(fasta_path, output_dir, template_path):
    # the first argument is the name of the script, so the second is the input file
    feature_dict = create_manual_feature(fasta_path, template_path)

    features_output_path = os.path.join(output_dir, 'features.pkl')
    with open(features_output_path, 'wb') as f:
        pickle.dump(feature_dict, f, protocol=4)




def main(argv):
    # Check for duplicate FASTA file names.
    fasta_names = [pathlib.Path(p).stem for p in FLAGS.fasta_paths]
    if len(fasta_names) != len(set(fasta_names)):
        raise ValueError('All FASTA paths must have a unique basename.')
    
    for i, fasta_path in enumerate(FLAGS.fasta_paths):
        fasta_name = fasta_names[i]

        # Create output directories.
        output_dir = os.path.join(FLAGS.output_dir, fasta_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        manual_feature(fasta_path, output_dir, FLAGS.template_path)


if __name__ == '__main__':
    flags.mark_flags_as_required([
      'fasta_paths',
      'output_dir',
      'template_path'
  ])
    app.run(main)
