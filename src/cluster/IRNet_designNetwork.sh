#!/bin/bash

#SBATCH --chdir=/scratch/nas/3/jparedes/IRNet/src
#SBATCH --job-name="IRNet_design"
#SBATCH --output=/scratch/nas/3/jparedes/IRNet/outputs/output-IRNet_design-%j-%N.out
#SBATCH --error=/scratch/nas/3/jparedes/IRNet/errors/error-IRNet_design-%j-%N.out

epsilon=1e-2
num_it=20
variance_threshold_ratio=1.5
signal_sparsity=150

python network_design_SST.py --design_large_networks -s ${signal_sparsity} -e ${epsilon} -n_it ${num_it} -vtr ${variance_threshold_ratio}
