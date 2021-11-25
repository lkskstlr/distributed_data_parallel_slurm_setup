#!/bin/bash

# This will set the env variables MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK
setup_str=$( python distributed_data_parallel_slurm_setup.py "$@" )
eval $setup_str

# TODO: You can simply call your DDP-enabled script here. See example.py for an example.
# Docs: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
python example.py