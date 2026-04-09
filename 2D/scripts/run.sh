DIRNAME="square10sep20num100"
SEPARATION=20
SIDELLENX=10
SIDELLENY=10
PADDING=20
LC_SMALL=1
LC_LARGE=5
ELEMENT_ORDER=2
N=100
echo ${N}

cd /home/ubuntu/cqednumerics/2D

python3 -u 2DTwoModesOpt.py \
    --plotdir="./allplots/${DIRNAME}/" \
    --separation=${SEPARATION} \
    --padding=${PADDING} \
    --lc_small=${LC_SMALL} \
    --sidelenX=${SIDELLENX} \
    --sidelenY=${SIDELLENY} \
    --N=${N} \
    --element_order=${ELEMENT_ORDER} \
    --opt_tol=0.001 \
    --opt_maxiter=500 \
    --lc_large=${LC_LARGE} \
    --full_lambda_y=True \
