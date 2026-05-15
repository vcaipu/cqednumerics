#!/bin/bash
#!/bin/bash

SCALE=1

DIRNAME="rabilondon/rl12"   
SEPARATION=$((20 * SCALE))
SIDELLENX=$((20 * SCALE))
SIDELLENY=$((20 * SCALE))
SIDELLENZ=$((20 * SCALE))
PADDING=$((40 * SCALE))
LC_SMALL=$((2 * SCALE))
LC_LARGE=$((5 * SCALE))
ELEMENT_ORDER=2


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
    --material=0.0033784 \
    --plotdir="./allplots/${DIRNAME}/" \
    --separation=${SEPARATION} \
    --padding=${PADDING} \
    --lc_small=${LC_SMALL} \
    --sidelenX=${SIDELLENX} \
    --sidelenY=${SIDELLENY} \
    --sidelenZ=${SIDELLENZ} \
    --element_order=${ELEMENT_ORDER} \
    --opt_tol=0.001 \
    --opt_maxiter=500 \
    --lc_large=${LC_LARGE} > "./allplots/${DIRNAME}/output.txt" 2>&1
