#!/bin/sh

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --account=pls0144
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task 4
##SBATCH --constraint=48core 
#SBATCH -J mnist
#SBATCH -o alazar-%j.out
#SBATCH --mail-type=END
#SBATCH --mail-user=alazar@ysu.edu

export NCCL_NET_GDR_LEVEL=PHB
export NCCL_P2P_LEVEL=NVL

##### Number of total processes 
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

module load cuda/11.8.0
source activate torch
srun --ntasks-per-node=2 --gpu_cmode=exclusive python src/3_mnist_pl.py
#--ntasks-per-node=1