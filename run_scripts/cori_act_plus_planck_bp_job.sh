#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
#  for TACC Stampede2 KNL nodes
#
#  *** Serial Job on Normal Queue ***
#
# Last revised: 20 Oct 2017
#
# Notes:
#
#  -- Copy/edit this script as desired. Launch by executing
#   "sbatch knl.serial.slurm" on a Stampede2 login node.
#
#  -- Serial codes run on a single node (upper case N = 1).
#    A serial code ignores the value of lower case n,
#    but slurm needs a plausible value to schedule the job.
#
#  -- For a good way to run multiple serial executables at the
#    same time, execute "module load launcher" followed
#    by "module help launcher".
#----------------------------------------------------
#SBATCH -J appbp      # Job name
#SBATCH -o appbp.o%j	  # Name of stdout output file
#SBATCH -e appbp.e%j	  # Name of stderr error file
# SBATCH -p skx-normal   # Queue (partition) name
#SBATCH -N 1      # Total # of nodes (must be 1 for serial)
#SBATCH -n 32        # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 48:00:00    # Run time (hh:mm:ss)
#SBATCH -C haswell
# module load gcc/10.1.0
# module load python/3.8-anaconda-2020.11
# module load gsl/2.5
# module load openmpi/4.0.2
export OMP_NUM_THREADS=16
# mpirun -np 4 /home1/08134/tg874332/.local/bin/cobaya-run /home1/08134/tg874332/bplike/run_scripts/act_extended_act_only_mcmc_classy_stampede.yml -f
# mpirun -np 4 /home1/08134/tg874332/.local/bin/cobaya-run /home1/08134/tg874332/bplike/run_scripts/act_extended_act_only_bp_mcmc_classy_stampede.yml -f
/global/homes/b/bb3028/.conda/envs/myenv/bin/mpirun -np 4 /global/homes/b/bb3028/.conda/envs/myenv/bin/cobaya-run /global/homes/b/bb3028/bplike/run_scripts/act_extended_act_plus_planck_bp_mcmc_classy_cori.yml -f
