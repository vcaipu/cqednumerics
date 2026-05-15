FILENAME="testrun"
KAPPA=3
XI=0.39e-10
SOURCE_X=0
SOURCE_Y=40
SIGMA=10
M=200
NUM_RABI_PERIODS=0.3
STEPS_PER_DRIVE_PERIOD=5

export JAX_ENABLE_X64=true

cd /scratch/gpfs/AROD/vc9839/finite-island-cqed/RabiLondon

python3 -u 2DRabiLondon.py \
    --savefile="./allplots/${FILENAME}.pkl" \
    --readfile="./../2D/allplots/rabilondonfine11/results.pkl" \
    --kappa=${KAPPA} \
    --xi=${XI} \
    --source_coord ${SOURCE_X} ${SOURCE_Y} \
    --sigma=${SIGMA} \
    --m=${M} \
    --num_rabi_periods=${NUM_RABI_PERIODS} \
    --steps_per_drive_period=${STEPS_PER_DRIVE_PERIOD} > "./allplots/${FILENAME}.log" 2>&1