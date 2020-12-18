#!/bin/bash
X#SBATCH --job-name=pc1-intro
#SBATCH --partition=s_hadoop
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=2:00
module purge
module load tools/python/3.7
module load mpi/intel/2019-Update5
srun python warming_stripes_mpi_distributed.py
