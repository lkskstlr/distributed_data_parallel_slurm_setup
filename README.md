# Pytorch Distributed Data Parallel (DDP) on a SLURM Cluster

**STILL WORK IN PROGRESS**

This repository contains files that enable the usage of DDP on a cluster managed with SLURM.

**Your workflow:**
* Integrate PyTorch DDP usage into your `train.py` (or similar) by following `example.py`, which is a slightly adapted example from [pytorch/examples](https://github.com/pytorch/examples/tree/master/distributed/ddp), and the [online docs](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html).
* Edit `distributed_data_parallel_slurm_run.bash` to call your script and not `example.py`.
* Edit `distributed_data_parallel_slurm_setup.sbatch` to adapt the SLURM launch parameters:
  * `--nodes=1`: Number of nodes
  * `--ntasks-per-node=X`: Number of tasks per node. Each task will have one GPU, which gives `nodes*X` total GPUs
  * `--gres=gpu:X,12G`: Should be the same as `ntasks-per-node`. Request the amount of VRAM per GPU that you need.
  * `--cpus-per-task=1`: Number of CPU cores per task. Usually a number larger than 1 is better, maybe 3 to 6.
  * `--mem=4G`: RAM per node. Set to `10*X` if one task needs `10G` of RAM.
  * `--time=00:01:00`: Maximal runtime of the job (will be killed afterwards).
  * `--output`: File to save the output to. See the `slurm-log` function defined below.
* Submit you job to the SLURM queue with `sbatch distributed_data_parallel_slurm_setup.sbatch`. Make sure that the correct python interpreter is in the path, e.g. by calling `conda activate my_env` before.


### Further Information

**SLURM pitfalls**:
* `sbatch` executes your script once ressources are available and will use the filesystem as it is at that point. If you call `sbatch` and then edit your code files you will run the edited code. This cannot be easily circumvented. The python interpreter loads all files on startup into RAM, which means that you can edit your code files after the job is running.
* All machines use a shared network filesystem, which means that if you change your data while a job is running it will use the changed data.

**SLURM-related functions:**
These functions are what I use, but they might not be optimal for you. If you use `slurm-log` you must call `mkdir -p $HOME/slurm/logs` once to generate the output directory. You can call `slurm-log` and it will `tail -f` the output of the output file with the newest timestamp. You can also call `slurm-log JOB_ID` and it will `tail -f` the output of the job with the id `JOB_ID`. The function `msq` is just an alias for `squeue` with more compact output. You have to put your username there.
```bash
function slurm-log() {
    if [ "$1" = "" ]
    then
        unset -v latest_slurm_log
        for file in "$HOME/slurm/logs/slurm-"*.out;
        do
            latest_slurm_log=$file
        done

        echo "tail $latest_slurm_log"
        tail -f -n+1 --retry --follow=name "$latest_slurm_log"
    else
        tail -f -n+1 --retry --follow=name "$HOME/slurm/logs/slurm-$1.out"
    fi
}

# TODO: Insert you SLURM user here
function msq () {
    squeue -u koestlel --format="%.7i %50j %.2t %.10M %.6D %R"
}
```