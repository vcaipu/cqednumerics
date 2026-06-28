#!/bin/bash
#SBATCH --job-name=RabiLondon        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1             # total number of tasks across all nodes
#SBATCH --mem-per-cpu=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=/scratch/gpfs/AROD/vc9839/finite-island-cqed/testouput.txt
#SBATCH --error=/scratch/gpfs/AROD/vc9839/finite-island-cqed/testerror.txt

module purge
cd /scratch/gpfs/AROD/vc9839/finite-island-cqed
source start.sh

cd ./RabiLondon/scripts
bash runRL.sh
