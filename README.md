# FoldCraft

Note: this project is **WORK IN PROGRESS**.

Implementation of AlphaFold in independent modules. AlphaFold became a sandbox in FoldCraft, and you can play with it as you wish.

## Installation
Using conda
```bash
conda create -n foldcraft python=3.10
conda install ipykernel   # recommendation
pip install ml-collections==0.1.1
conda install numpy jax absl-py
conda install dm-haiku dm-tree -c conda-forge
conda install tensorflow-cpu -c conda-forge
conda install biopython -c conda-forge
```


## Installed package list
- python=3.10
- ml-collections==0.1.1   # only pypi
- absl-py
- jax
- dm-haiku
- dm-tree
- numpy
- biopython=1.82
- tensorflow=2.15.0    # required in alphafold.model.features
- [?]pandas   
- [?]scipy
- [?]immutabledict


## Run

**Step 1**: Create empty MSA feature (input feature of alphafold)
```bash
python parafold/create_empty_feature.py --fasta_paths input/test.fasta --output_dir output
```

**Step 2**: Run FoldCraft
```bash
python run_foldcraft.py
```




