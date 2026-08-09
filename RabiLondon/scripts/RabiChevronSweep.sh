#!/usr/bin/env bash

# Script to call 3DRabiLondon.py for a sweep of detunings between -detuning_max and detuning_max

# Resolve paths relative to this script so defaults work from any cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ==== USER INPUTS ====
INPUT_PKL_DEFAULT="$REPO_ROOT/3D/allplots/sepsweep3/sep15/results.pkl"    # Or modify as needed
OUTPUT_PKL_DIR_DEFAULT="$REPO_ROOT/RabiLondon/allplots/rabichevron1"      # Will save one file per detuning
DETUNING_MAX_DEFAULT=1e-8                                        # Change as needed
NUM_STEPS=11                                                     # Odd prefered so 0 is included (center point)

# Read arg1 as input picke, else go with default
INPUT_PKL=${1:-$INPUT_PKL_DEFAULT}
OUTPUT_PKL_DIR=${2:-$OUTPUT_PKL_DIR_DEFAULT}
DETUNING_MAX=${3:-$DETUNING_MAX_DEFAULT}


if [ ! -f "$INPUT_PKL" ]; then
    echo "Input pickle file does not exist: $INPUT_PKL"
    exit 1
fi

mkdir -p "$OUTPUT_PKL_DIR"

# Create sequence of detunings from -DETUNING_MAX to DETUNING_MAX, inclusive, NUM_STEPS total
detuning_list=()
for i in $(seq 0 $((NUM_STEPS-1))); do
    frac=$(awk -v n=$i -v N=$((NUM_STEPS-1)) 'BEGIN { printf "%.12f", (n - N/2.0)/(N/2.0) }')
    detuning=$(awk -v dmax=$DETUNING_MAX -v f=$frac 'BEGIN { printf "%.12g", dmax*f }')
    detuning_list+=("$detuning")
done

echo "Sweeping detuning values: ${detuning_list[@]}"

for detuning in "${detuning_list[@]}"
do
    output_file="$OUTPUT_PKL_DIR/sweep_detuning_${detuning}.pkl"
    echo "Running detuning = $detuning"
    python3 "$REPO_ROOT/RabiLondon/3DRabiLondon.py" \
        --input-pickle-file "$INPUT_PKL" \
        --output-pickle-file "$output_file" \
        --detuning-mult="$detuning"
done

echo "Sweep finished. Output files saved in: $OUTPUT_PKL_DIR"