DIRNAME="rabilondonscaletest26"
SCALE=20
# Bash arithmetic: $(( ... )); ** is power (^ is XOR, not exponentiation)
N=$((100 * SCALE*2))
n=50
SEPARATION=$((20 * SCALE * 4))
SIDELLEN=$((20 * SCALE))

LC_LARGE=$((5 * SCALE))
LC_SMALL=$((1 * SCALE))
ELEMENT_ORDER=2
PADDING=$((30 * SCALE))
GEOMETRY="rect"
INNER_DIMX=$((10 * SCALE))
INNER_DIMY=$((10 * SCALE))
OPT_TOL_NEW=$(awk "BEGIN {printf \"%.8f\", 0.1 * $SCALE}") 
OPT_MAXITER=1500

# 1. Run NEW Script
cd /scratch/gpfs/AROD/vc9839/finite-island-cqed/2D
mkdir -p "./allplots/${DIRNAME}/"
python3 -u 2DTwoModesOpt.py \
    --plotdir="./allplots/${DIRNAME}/" \
    --separation=${SEPARATION} \
    --sidelenX=${SIDELLEN} \
    --sidelenY=${SIDELLEN} \
    --N=${N} \
    --n=${n} \
    --geometry=${GEOMETRY} \
    --inner_dimX=${INNER_DIMX} \
    --inner_dimY=${INNER_DIMY} \
    --padding=${PADDING} \
    --lc_small=${LC_SMALL} \
    --lc_large=${LC_LARGE} \
    --element_order=${ELEMENT_ORDER} \
    --opt_tol=${OPT_TOL_NEW} \
    --opt_maxiter=${OPT_MAXITER} \
    --no-full_lambda_y > "./allplots/${DIRNAME}/output.txt" 2>&1