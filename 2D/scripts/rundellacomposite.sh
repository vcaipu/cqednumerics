DIRNAME="rabilondoncomposite2"
N=100
n=50
SEPARATION=20
SIDELLENX=20
SIDELLENY=5
SIDELLEN2X=80
SIDELLEN2Y=80
LC_LARGE=10
LC_SMALL=2
ELEMENT_ORDER=2
PADDING=30
GEOMETRY="composite"
INNER_DIM=5
OPT_TOL_NEW=5e-4 # Remember it is full Hamiltonian, vs old is per particle Hamiltonian
OPT_MAXITER=500

# 1. Run NEW Script
cd /scratch/gpfs/AROD/vc9839/finite-island-cqed/2D
mkdir -p "./allplots/${DIRNAME}/"
python3 -u 2DTwoModesOpt.py \
    --plotdir="./allplots/${DIRNAME}/" \
    --separation=${SEPARATION} \
    --sidelenX=${SIDELLENX} \
    --sidelenY=${SIDELLENY} \
    --sidelen2X=${SIDELLEN2X} \
    --sidelen2Y=${SIDELLEN2Y} \
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