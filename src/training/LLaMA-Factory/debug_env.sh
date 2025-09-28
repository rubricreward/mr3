#!/bin/bash
#SBATCH --job-name=debug-env
#SBATCH --partition=compute_full_node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -t 00:05:00

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS

srun bash -c '
echo "=== PROCESS INFO ==="
echo "Hostname: $(hostname)"
echo "SLURM_PROCID (-> RANK): $SLURM_PROCID"
echo "SLURM_LOCALID (-> LOCAL_RANK): $SLURM_LOCALID"
echo "SLURM_NODEID: $SLURM_NODEID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "RANK: $SLURM_PROCID"
echo "LOCAL_RANK: $SLURM_LOCALID"
echo
'
