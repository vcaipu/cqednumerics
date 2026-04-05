DIRNAME="Run $(date +'%Y-%m-%d %H:%M:%S')"
SEPARATION=10
SIDELLENX=10
SIDELLENY=20
PADDING=25
LC_SMALL=0.6
LC_LARGE=5
ELEMENT_ORDER=2
N=200
echo ${N}

cd /Users/vincentcai/Desktop/cqednumerics/2D

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
    --mesh_file="./meshes/custommesh.msh"