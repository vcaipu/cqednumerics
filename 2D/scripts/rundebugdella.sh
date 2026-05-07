DIRNAME="debugdella2"
N=70
n=50
SEPARATION=30
SIDELLEN=20

LC_LARGE=5
LC_SMALL=0.8
ELEMENT_ORDER=2
PADDING=25
GEOMETRY="rect"
INNER_DIM=30
FULL_LAMBDA_Y=False
OPT_TOL=1e-4
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
    --opt_tol=${OPT_TOL} \
    --opt_maxiter=${OPT_MAXITER} \
    --full_lambda_y=${FULL_LAMBDA_Y}