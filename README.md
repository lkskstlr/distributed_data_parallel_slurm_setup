# Pytorch Distributed Data Parallel (DDP) on a SLURM Cluster

This repository cotains files that enable the usage of DDP on a cluster managed with SLURM.

**Your Workflow:**
* Integrate PyTorch DDP usage into your `train.py` (or similar) by following `example.py`, which is a slightly adapted example from [pytorch/examples](https://github.com/pytorch/examples), and/or the online docs. This repository is not a HowTo for DDP.
* Edit `distributed_data_parallel_slurm_run.bash` to call your script and not `example.py`.
* Edit `distributed_data_parallel_slurm_setup.sbatch` to adapt the SLURM launch parameters, e.g. number of nodes or GPU VRAM. This repository is not a HowTo for SLURM.
* Submit you job to the SLURM queue with `sbatch distributed_data_parallel_slurm_setup.sbatch`. Make sure that the correct python interpreter is in the path, e.g. by calling `conda activate my_env` before).


### Further Information

**SLURM Pitfalls**:
* `sbatch` executes your script once ressources are available and will use the filesystem as it is at that point. If you call `sbatch` and then edit your code files you will run the edited code. This cannot be easily circumvented. The python interpreter loads all files on startup into RAM, which means that you can edit your code files after the job is running.
* All machines use a shared network filesystem, which means that if you change your data while a job is running it will use the changed data.