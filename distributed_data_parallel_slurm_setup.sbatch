#!/bin/bash -l

# sbatch documentation: https://slurm.schedmd.com/sbatch.html

# SLURM SUBMIT SCRIPT
# Rules:
#   Set --nodes to the number of nodes
#   Set --gres=gpu:X,VRAM:24G and --ntasks-per-node=X to the same number
#   Set --mem to about 10*X (e.g. 20G for two GPUs per node, if your job needs 10GB of RAM per GPU)
#   This will give nodes*X total GPUs

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:01:00
#SBATCH --output=/usr/wiss/%u/slurm/logs/slurm-%j.out

# --- Setup In Master ---
# Master will be the one running this bash script (SLURM runs this only once)
# Get hostname and port on first node first process
# For the port see: https://unix.stackexchange.com/questions/55913/whats-the-easiest-way-to-find-an-unused-local-port
master_addr=$(hostname -i)
master_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')


# --- Call the Script which the User will Edit ---
# With srun this will be run on all nodes for all processes
srun --gres=gpu:1 ./distributed_data_parallel_slurm_run.bash --master_addr $master_addr --master_port $master_port
