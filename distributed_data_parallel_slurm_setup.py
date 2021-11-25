import os
import socket
import argparse

# This script should run for python 2 and 3 without any packages.
# It targets linux because all SLURM clusters I know run Linux.

# It will output everything as a bash command that can then be used within the sbatch script

parser = argparse.ArgumentParser(description="Setup Pytorch Distributed Data Parallel (DDP) on a cluster managed by SLURM.")
parser.add_argument("--master_addr", required=True, type=str, help="Corresponds to MASTER_ADDR")
parser.add_argument("--master_port", required=True, type=int, help="Corresponds to MASTER_PORT")

if __name__ == "__main__":
    args = parser.parse_args()
    
    # --- Master Address and Port ---
    output = "export MASTER_ADDR='{}' && export MASTER_PORT='{}'".format(args.master_addr, str(args.master_port))

    # --- Compute world size ---
    # We assume one process per GPU, thus the world size = num_nodes*tasks_per_node
    num_nodes = int(os.environ["SLURM_JOB_NUM_NODES"])
    tasks_per_node = int(os.environ["SLURM_NTASKS_PER_NODE"])
    world_size = num_nodes * tasks_per_node
    output += " && export WORLD_SIZE='{}'".format(world_size)

    # --- Set world rank ---
    world_rank = os.environ['SLURM_PROCID']
    output += " && export RANK='{}'".format(world_rank, world_rank)

    print(output)
