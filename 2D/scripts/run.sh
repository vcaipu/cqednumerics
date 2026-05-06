<<<<<<< HEAD
DIRNAME="testlargeisland3"
SEPARATION=300
SIDELLENX=200
SIDELLENY=50
PADDING=0
LC_SMALL=3
LC_LARGE=15
=======
DIRNAME="rabilondonfinal4"
SEPARATION=20
SIDELLENX=10
SIDELLENY=10
SIDELLEN2X=100
SIDELLEN2Y=100  # Set to 0 for square geometry
PADDING=20
LC_SMALL=1
LC_LARGE=2
>>>>>>> awsbranch
ELEMENT_ORDER=2
GEOMETRY="rect"
INNER_DIM=30
N=100
echo ${N}

cd /home/ubuntu/cqednumerics/2D

mkdir -p "./allplots/${DIRNAME}/"
python3 -u 2DTwoModesOpt.py \
    --plotdir="./allplots/${DIRNAME}/" \
    --separation=${SEPARATION} \
    --padding=${PADDING} \
    --lc_small=${LC_SMALL} \
    --sidelenX=${SIDELLENX} \
    --sidelenY=${SIDELLENY} \
    --sidelen2X=${SIDELLEN2X} \
    --sidelen2Y=${SIDELLEN2Y} \
    --geometry=${GEOMETRY} \
    --inner_dim=${INNER_DIM} \
    --N=${N} \
    --element_order=${ELEMENT_ORDER} \
    --opt_tol=0.1 \
    --opt_maxiter=500 \
    --lc_large=${LC_LARGE} \
    --full_lambda_y=True > "./allplots/${DIRNAME}/output.txt"
