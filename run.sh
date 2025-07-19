#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Experiment Configuration ---
DATASETS=("busi" "bus_uc" "busbra" "bus_uclm" "yap2018")
MODELS=("UNet" "AttUNet" "UNetplus" "UNet3plus" "UNeXt" "CMUNet" "CMUNeXt")
CONFIG_FILE="config.json"
NUM_FOLDS=5

# --- Main Experiment Loop ---
for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    
    echo -e "\n\n======================================================="
    echo "  STARTING EXPERIMENT: DATASET=${dataset} | MODEL=${model}"
    echo "======================================================="

    RUN_ID="${dataset}_${model}"
    CHECKPOINT_BASE_DIR="checkpoints"
    
    # --- Check if training is already completed ---
    NUM_CHECKPOINTS=0
    if [ -d "$CHECKPOINT_BASE_DIR" ]; then
        NUM_CHECKPOINTS=$(find "$CHECKPOINT_BASE_DIR" -name "${RUN_ID}_fold*_best.pth" | wc -l)
    fi

    # --- Training Phase (Conditional) ---
    if [ "$NUM_CHECKPOINTS" -eq "$NUM_FOLDS" ]; then
      echo "All ${NUM_FOLDS} checkpoints found for ${RUN_ID}. Skipping training."
    else
      echo "Checkpoints not found or incomplete (${NUM_CHECKPOINTS}/${NUM_FOLDS}) for ${RUN_ID}. Starting training."
      python train.py -c ${CONFIG_FILE} \
                      --name "${RUN_ID}" \
                      --datasets "${dataset}" \
                      --model ${model}
    fi

    # --- Testing & Aggregation Phase ---
    echo -e "\n--- Phase: Testing and Aggregating Results ---"
    
    RESULTS_CSV="results_${RUN_ID}.csv"
    # --- FIX: Add GFLOPs and Params to the CSV header ---
    echo "PA,DSC,HD95,IoU,GFLOPs,Params" > ${RESULTS_CSV}

    for fold in $(seq 1 ${NUM_FOLDS}); do
      CHECKPOINT_PATH="${CHECKPOINT_BASE_DIR}/${RUN_ID}_fold${fold}_best.pth"
      
      if [ -f "$CHECKPOINT_PATH" ]; then
        echo "Testing checkpoint: $CHECKPOINT_PATH"
        TEST_OUTPUT=$(python test.py -r "$CHECKPOINT_PATH")
        
        # Parse all metrics from the output
        PA=$(echo "$TEST_OUTPUT" | grep "PA:" | cut -d':' -f2 | xargs)
        DSC=$(echo "$TEST_OUTPUT" | grep "DSC:" | cut -d':' -f2 | xargs)
        HD95=$(echo "$TEST_OUTPUT" | grep "HD95:" | cut -d':' -f2 | xargs)
        IOU=$(echo "$TEST_OUTPUT" | grep "IoU:" | cut -d':' -f2 | xargs)
        # --- FIX: Parse GFLOPs and Params ---
        GFLOPS=$(echo "$TEST_OUTPUT" | grep "GFLOPs:" | cut -d':' -f2 | xargs)
        PARAMS=$(echo "$TEST_OUTPUT" | grep "Params:" | cut -d':' -f2 | xargs)
        
        # --- FIX: Write all parsed values to the CSV ---
        echo "${PA},${DSC},${HD95},${IOU},${GFLOPS},${PARAMS}" >> ${RESULTS_CSV}
      else
        echo "Error: Checkpoint for fold ${fold} not found at ${CHECKPOINT_PATH}!"
      fi
    done

    # --- Calculate and Print Final Statistics ---
    if [ -f "aggregate_results.py" ]; then
        python aggregate_results.py ${RESULTS_CSV}
    else
        echo "Warning: aggregate_results.py not found. Skipping statistics calculation."
    fi

    # --- FIX: Comment out the line that removes the CSV file ---
    # rm ${RESULTS_CSV}
    echo "--- EXPERIMENT FINISHED: DATASET=${dataset} | MODEL=${model} ---"

  done
done

echo -e "\n\n======================================================="
echo "  ALL EXPERIMENTS COMPLETED"
echo "======================================================="
