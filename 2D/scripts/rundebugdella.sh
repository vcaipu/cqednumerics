DIRNAME="debugdella9"
N=100
n=50
SEPARATION=20
SIDELLEN=10

LC_LARGE=1.3
LC_SMALL=1.3
ELEMENT_ORDER=1
PADDING=20
GEOMETRY="rect"
INNER_DIM=4
OPT_TOL_OLD=5e-6
OPT_TOL_NEW=5e-4 # Remember it is full Hamiltonian, vs old is per particle Hamiltonian
OPT_MAXITER=1000

# 1. Run OLD Script
cd /scratch/gpfs/AROD/vc9839/finite-island-cqed/2D/archives
mkdir -p "./../allplots/${DIRNAME}-OLD/"
python3 -u 2DZeroPoint.py \
    --plotdir="./../allplots/${DIRNAME}-OLD/" \
    --sidelen=${SIDELLEN} \
    --separation=${SEPARATION} \
    --N=${N} \
    --n=${n} \
    --opt_tol=${OPT_TOL_OLD} \
    --opt_maxiter=${OPT_MAXITER} \


# 2. Run NEW Script
cd /scratch/gpfs/AROD/vc9839/finite-island-cqed/2D
mkdir -p "./allplots/${DIRNAME}-NEW/"
python3 -u 2DTwoModesOpt.py \
    --plotdir="./allplots/${DIRNAME}-NEW/" \
    --separation=${SEPARATION} \
    --sidelenX=${SIDELLEN} \
    --sidelenY=${SIDELLEN} \
    --N=${N} \
    --n=${n} \
    --geometry=${GEOMETRY} \
    --inner_dim=${INNER_DIM} \
    --padding=${PADDING} \
    --lc_small=${LC_SMALL} \
    --lc_large=${LC_LARGE} \
    --element_order=${ELEMENT_ORDER} \
    --opt_tol=${OPT_TOL_NEW} \
    --opt_maxiter=${OPT_MAXITER} \
    --no-full_lambda_y