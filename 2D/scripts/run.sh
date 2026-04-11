DIRNAME="testlargeisland3"
SEPARATION=300
SIDELLENX=200
SIDELLENY=50
PADDING=200
LC_SMALL=3
LC_LARGE=15
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
