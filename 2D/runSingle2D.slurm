#!/bin/bash
#SBATCH --job-name=TwoModes3D        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1             # total number of tasks across all nodes
#SBATCH --gres=gpu:1       # number of gpus per node
#SBATCH --time=4:00:00          # total run time limit (HH:MM:SS)
#SBATCH --cpus-per-task=4           # 4 cores for mesh generation/linalg
#SBATCH --mem=128G           # Request 128GB of total RAM for the job
#SBATCH --output=/scratch/gpfs/AROD/vc9839/finite-island-cqed/currentoutput.txt
#SBATCH --error=/scratch/gpfs/AROD/vc9839/finite-island-cqed/currenterror.txt


########################################################
# HOW TO 
########################################################

# Command line arguments usage:
#   $1: DIRNAME      (Name of results directory, default: "Run <current date/time>")
#   $2: MATERIAL     (Material parameter, default: 0.106)
#   $3: SEPARATION   (Separation distance, default: 200)
#   $4: GRIDLEN      (Grid length, default: 250)
#   $5: LC_SMALL     (Small characteristic length, default: 0.8)
#   $6: LC_LARGE     (Large characteristic length, default: 5)

# For example:
#   sbatch Single.slurm "TestDir" 0.12 300 400 1.0 10

########################################################
# SCRIPT STARTS BELOW
########################################################

# Accessing the arguments
DIRNAME=${1:-"Run $(date +'%Y-%m-%d %H:%M:%S')"}
SEPARATION=${2:-200}
SIDELLENX=${3:-20}
SIDELLENY=${4:-20}
PADDING=${5:-10}
LC_SMALL=${6:-0.8}
LC_LARGE=${7:-5}
ELEMENT_ORDER=${8:-1}
N=${9:-70}


echo "Running with the following parameters:"
echo "DIRNAME: ${DIRNAME}"
echo "SEPARATION: ${SEPARATION}"
echo "SIDELLENX: ${SIDELLENX}"
echo "SIDELLENY: ${SIDELLENY}"
echo "PADDING: ${PADDING}"
echo "LC_SMALL: ${LC_SMALL}"
echo "LC_LARGE: ${LC_LARGE}"

# Set up
module purge
cd /scratch/gpfs/AROD/vc9839/finite-island-cqed
source start.sh
cd ./3D

# Create the directory, need to do this at the start because you're piping the output to a file in this directory. 
mkdir -p "./allplots/${DIRNAME}/"

# Run the script (python3 -u = unbuffered so log order matches execution)
python3 -u 2DTwoModesOpt.py \
    --plotdir="./allplots/${DIRNAME}/" \
    --separation=${SEPARATION} \
    --padding=${PADDING} \
    --lc_small=${LC_SMALL} \
    --sidelenX=${SIDELLENX} \
    --sidelenY=${SIDELLENY} \
    --N=${N} \
    --element_order=${ELEMENT_ORDER} \
    --opt_tol=0.001 \
    --opt_maxiter=500 \
    --lc_large=${LC_LARGE} > "./allplots/${DIRNAME}/output.txt" 2>&1 \
    --full_lambda_y=True


# --mesh_file="./sweepareamesh.msh" \