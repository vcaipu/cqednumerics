#!/bin/bash
#!/bin/bash

DIRNAME="heightsweep/height${1}"
SEPARATION=1
SIDELLENX=10
SIDELLENY=20
SIDELLENZ=${1}
PADDING=20
LC_SMALL=1
LC_LARGE=15
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
cd ~/cqednumerics
source startaws.sh
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
    --opt_maxiter=1000 \
    --lc_large=${LC_LARGE} > "./allplots/${DIRNAME}/output.txt" 2>&1
