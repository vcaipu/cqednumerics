#!/bin/bash
#SBATCH --job-name=SweepSeparations3D        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1             # total number of tasks across all nodes
#SBATCH --gres=gpu:1       # number of gpus per node
#SBATCH --mem-per-cpu=30G
#SBATCH --cpus-per-task=2
#SBATCH --time=3:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=/scratch/gpfs/AROD/vc9839/finite-island-cqed/testouput.txt
#SBATCH --error=/scratch/gpfs/AROD/vc9839/finite-island-cqed/testerror.txt

module purge
cd /scratch/gpfs/AROD/vc9839/finite-island-cqed
source start.sh

cd ./3D
python3 ./3Dsweepseparations.py 