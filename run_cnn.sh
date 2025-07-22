#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- CNN Experiment Configuration ---
DATASETS=("busi" "bus_uc" "busbra" "bus_uclm" "yap2018")
CNN_MODELS=("UKAN")
# CNN_MODELS=("UNet" "AttUNet" "UNetplus" "UNet3plus" "UNeXt" "CMUNet" "CMUNeXt")
CONFIG_FILE="config.json"
NUM_FOLDS=5

# --- Main Experiment Loop for CNNs ---
for dataset in "${DATASETS[@]}"; do
  for model in "${CNN_MODELS[@]}"; do
    
    echo -e "\n\n======================================================="
    echo "  STARTING CNN EXPERIMENT: DATASET=${dataset} | MODEL=${model}"
    echo "======================================================="

    RUN_ID="lovasz_${dataset}_${model}"
    CHECKPOINT_BASE_DIR="checkpoints"
    
    # Check if training is already completed
    NUM_CHECKPOINTS=0
    if [ -d "$CHECKPOINT_BASE_DIR" ]; then
        NUM_CHECKPOINTS=$(find "$CHECKPOINT_BASE_DIR" -name "${RUN_ID}_fold*_best.pth" | wc -l)
    fi

    # Training Phase (Conditional)
    if [ "$NUM_CHECKPOINTS" -eq "$NUM_FOLDS" ]; then
      echo "All ${NUM_FOLDS} checkpoints found for ${RUN_ID}. Skipping training."
    else
      echo "Starting training for ${RUN_ID}."
      python train.py -c ${CONFIG_FILE} \
                      --name "${RUN_ID}" \
                      --datasets "${dataset}" \
                      --model ${model}
    fi

    # Testing & Aggregation Phase
    echo -e "\n--- Testing and Aggregating Results for ${RUN_ID} ---"
    
    RESULTS_DIR="results"
    mkdir -p ${RESULTS_DIR}
    RESULTS_CSV="${RESULTS_DIR}/lovasz_results_${RUN_ID}.csv"
    echo "PA,DSC,HD95,IoU,GFLOPs,Params" > ${RESULTS_CSV}

    for fold in $(seq 1 ${NUM_FOLDS}); do
      CHECKPOINT_PATH="${CHECKPOINT_BASE_DIR}/${RUN_ID}_fold${fold}_best.pth"
      
      if [ -f "$CHECKPOINT_PATH" ]; then
        TEST_OUTPUT=$(python test.py -r "$CHECKPOINT_PATH")
        
        PA=$(echo "$TEST_OUTPUT" | grep "PA:" | cut -d':' -f2 | xargs)
        DSC=$(echo "$TEST_OUTPUT" | grep "DSC:" | cut -d':' -f2 | xargs)
        HD95=$(echo "$TEST_OUTPUT" | grep "HD95:" | cut -d':' -f2 | xargs)
        IOU=$(echo "$TEST_OUTPUT" | grep "IoU:" | cut -d':' -f2 | xargs)
        GFLOPS=$(echo "$TEST_OUTPUT" | grep "GFLOPs:" | cut -d':' -f2 | xargs)
        PARAMS=$(echo "$TEST_OUTPUT" | grep "Params:" | cut -d':' -f2 | xargs)
        
        echo "${PA},${DSC},${HD95},${IOU},${GFLOPS},${PARAMS}" >> ${RESULTS_CSV}
      else
        echo "Error: Checkpoint for fold ${fold} not found at ${CHECKPOINT_PATH}!"
      fi
    done

    echo "--- CNN EXPERIMENT FINISHED: ${RUN_ID} ---"
  done
done

echo -e "\n\n======================================================="
echo "  ALL CNN EXPERIMENTS COMPLETED"
echo "======================================================="

