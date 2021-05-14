#!/bin/bash
#SBATCH --account=hill # The account name for the job.
#SBATCH -N 1
#SBATCH -q regular
#SBATCH -J ao
#SBATCH --mail-user=bb3028@columbia.edu
#SBATCH --mail-type=ALL
#SBATCH -t 120:00:00

export OMP_NUM_THREADS=12
/moto/home/bb3028/.conda/envs/myenv/bin/mpirun -np 4 /moto/home/bb3028/.conda/envs/myenv/bin/cobaya-run /moto/home/bb3028/bplike/run_scripts/act_extended_act_only_mcmc_classy_mcmc.yml -f
# mpirun -np 4 /home1/08134/tg874332/.local/bin/cobaya-run /home1/08134/tg874332/bplike/run_scripts/act_extended_act_only_bp_mcmc_classy_stampede.yml -f
# mpirun -np 4 /home1/08134/tg874332/.local/bin/cobaya-run /home1/08134/tg874332/bplike/run_scripts/act_extended_act_plus_planck_bp_mcmc_classy_stampede.yml -f
