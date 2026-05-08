DIRNAME="rabilondonfine11"
N=100
n=50
SEPARATION=20
SIDELLEN=40

LC_LARGE=5
LC_SMALL=1
ELEMENT_ORDER=2
PADDING=30
GEOMETRY="rect"
INNER_DIMX=5
INNER_DIMY=5
OPT_TOL_NEW=5e-4 # Remember it is full Hamiltonian, vs old is per particle Hamiltonian
OPT_MAXITER=500

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
    --no-full_lambda_y