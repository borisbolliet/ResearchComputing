#!/usr/bin/env bash

# Benchmark info
echo "TIMING - Starting main script at: $(date)"

# Set working directory to home directory
cd "${HOME}"

#
# Start Jupyter Notebook Server
#

# Purge the module environment to avoid conflicts
module purge

# Load the require modules
# module load jupyterlab/3
module load rhel8/default-icl
module load cfitsio-3.410-intel-17.0.4-4qrgkot intel/bundles/complib/2020.4 gsl-2.3-intel-17.0.4-ewbwjes fftw/intel/64/3.3.3

source ~/.defaultEnv/bin/activate

which jupyter

# List loaded modules
module list

# Benchmark info
echo "TIMING - Starting jupyter at: $(date)"

# Launch the Jupyter Notebook Server
set -x
jupyter lab --config="${CONFIG_FILE}" 
