#!/bin/bash
#SBATCH --job-name=warming-stripes-mpi
#SBATCH --partition=s_hadoop
#SBATCH --mem-per-cpu=32000
#SBATCH --nodes=15
#SBATCH --ntasks-per-node=4
#SBATCH --time=20:00
module purge
module load tools/python/3.7
module load mpi/intel/2019-Update5
srun python /home/vi46six/warming_stripes_spark/scripts/distributed/warming_stripes_mpi_distributed.py

