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
