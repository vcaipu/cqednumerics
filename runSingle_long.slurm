#!/bin/bash
# Same as runSingle.slurm but with 2 hr time limit for tall islands (e.g. height 40).
# Use for height sweeps so jobs land in the >=62 min QOS and can run in parallel.
#SBATCH --job-name=TwoModes3D_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --output=/scratch/gpfs/AROD/vc9839/finite-island-cqed/currentoutput.txt
#SBATCH --error=/scratch/gpfs/AROD/vc9839/finite-island-cqed/currenterror.txt

DIRNAME=${1:-"Run $(date +'%Y-%m-%d %H:%M:%S')"}
SEPARATION=${2:-200}
SIDELLENX=${3:-20}
SIDELLENY=${4:-20}
SIDELLENZ=${5:-20}
PADDING=${6:-10}
LC_SMALL=${7:-0.8}
LC_LARGE=${8:-5}
ELEMENT_ORDER=${9:-1}

echo "Running with the following parameters:"
echo "DIRNAME: ${DIRNAME}"
echo "SEPARATION: ${SEPARATION}"
echo "SIDELLENX: ${SIDELLENX}"
echo "SIDELLENY: ${SIDELLENY}"
echo "SIDELLENZ: ${SIDELLENZ}"
echo "PADDING: ${PADDING}"
echo "LC_SMALL: ${LC_SMALL}"
echo "LC_LARGE: ${LC_LARGE}"

module purge
cd /scratch/gpfs/AROD/vc9839/finite-island-cqed
source start.sh
cd ./3D

mkdir -p "./allplots/${DIRNAME}/"

python3 -u 3DTwoModesOpt.py \
    --material=0.0106 \
    --plotdir="./allplots/${DIRNAME}/" \
    --separation=${SEPARATION} \
    --padding=${PADDING} \
    --lc_small=${LC_SMALL} \
    --sidelenX=${SIDELLENX} \
    --sidelenY=${SIDELLENY} \
    --sidelenZ=${SIDELLENZ} \
    --element_order=${ELEMENT_ORDER} \
    --lc_large=${LC_LARGE} > "./allplots/${DIRNAME}/output.txt" 2>&1
