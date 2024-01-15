# FoldCraft

Note: this project is **WORK IN PROGRESS**.

Implementation of AlphaFold in independent modules. AlphaFold became a sandbox in FoldCraft, and you can play with it as you wish.

## Installation

### CPU installation

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
Or you can just (Not ready for now):
```bash
conda env create -f environment/environment_cpu.yml
```

### GPU installation

Using NVIDIA A100 as example

```bash
conda create -n foldcraft python=3.10
conda install jax absl-py ipykernel dm-haiku dm-tree biopython -c conda-forge
conda install tensorflow-cpu -c conda-forge
pip install ml-collections==0.1.1
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Or you can just:
```bash
conda env create -f environment/environment_gpu.yml
```

### Apple silicon installation (GPU? No, it's still CPU)

Apple GPU might be supported in the future. (wait for jax-metal updates)

```bash
conda install ipykernel matplotlib
conda install jax absl-py -c conda-forge
conda install dm-haiku dm-tree -c conda-forge
conda install biopython -c conda-forge
conda install ml-collections -c conda-forge
conda install tensorflow-cpu -c conda-forge
# pip install jax-metal
```

In this environment, `dm-haiku` and `tensorflow-cpu` can bring in a lot of conflicts

```bash
conda env create -f environment/environment_m1.yml
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
- [?]immutabledict


## Run

**Step 1**: Create empty MSA feature (input feature of alphafold)
```bash
python foldcraft/parafold/create_empty_feature.py --fasta_paths input/test.fasta --output_dir output
```

**Step 2**: Run FoldCraft
```bash
python run_foldcraft.py
```



## Reference

- Jax installation: [link](https://jax.readthedocs.io/en/latest/installation.html)



