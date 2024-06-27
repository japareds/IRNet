#!/bin/bash

#SBATCH --chdir=/scratch/nas/3/jparedes/IRNet/src
#SBATCH --job-name="IRNet_sparsity"
#SBATCH --output=/scratch/nas/3/jparedes/IRNet/outputs/output-IRNet_sparsity-%j-%N.out
#SBATCH --error=/scratch/nas/3/jparedes/IRNet/errors/error-IRNet_sparsity-%j-%N.out

n_locations=1039
python network_design_SST.py --determine_sparsity --n_locations ${n_locations}
