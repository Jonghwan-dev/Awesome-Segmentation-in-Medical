#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# Define which pre-trained models to use as a source. Format: "dataset_model"
PRETRAINED_RUNS=(
    "busi_UNet"
    "busi_AttUNet"
    "busi_SwinUnet"
)

# Define which datasets to fine-tune on.
TARGET_DATASETS=("yap" "busbra")

# Fine-tuning Hyperparameters
CONFIG_FILE="config.json"
NUM_FOLDS=5
FT_EPOCHS=100
FT_LR=0.0001
FT_PATIENCE=20
FT_BATCH_SIZE=16
FT_FREEZE_MODE="encoder" # 'none' to train all layers, 'encoder' to freeze encoder

# --- Main Transfer Learning Loop ---
for source_run in "${PRETRAINED_RUNS[@]}"; do
    SOURCE_CHECKPOINT="checkpoints/${source_run}_fold1_best.pth"

    if [ ! -f "$SOURCE_CHECKPOINT" ]; then
        echo "Warning: Source checkpoint not found at ${SOURCE_CHECKPOINT}. Skipping."
        continue
    fi

    for target_dataset in "${TARGET_DATASETS[@]}"; do
        
        MODEL_NAME=$(echo ${source_run} | cut -d'_' -f2-)
        SOURCE_DATASET_NAME=$(echo ${source_run} | cut -d'_' -f1)
        RUN_ID="${SOURCE_DATASET_NAME}_${MODEL_NAME}_to_${target_dataset}_${FT_FREEZE_MODE}Frozen"

        echo -e "\n\n======================================================="
        echo "  STARTING TRANSFER EXPERIMENT: ${RUN_ID}"
        echo "  Source Checkpoint: ${SOURCE_CHECKPOINT}"
        echo "======================================================="

        CHECKPOINT_BASE_DIR="checkpoints"
        
        NUM_CHECKPOINTS=0
        if [ -d "$CHECKPOINT_BASE_DIR" ]; then
            NUM_CHECKPOINTS=$(find "$CHECKPOINT_BASE_DIR" -name "${RUN_ID}_fold*_best.pth" | wc -l)
        fi

        # Fine-tuning Phase
        if [ "$NUM_CHECKPOINTS" -eq "$NUM_FOLDS" ]; then
            echo "All ${NUM_FOLDS} checkpoints found for ${RUN_ID}. Skipping fine-tuning."
        else
            echo "Starting fine-tuning for ${RUN_ID}."
            python train.py -c ${CONFIG_FILE} \
                --transfer-from "${SOURCE_CHECKPOINT}" \
                --name "${RUN_ID}" \
                --model "${MODEL_NAME}" \
                --datasets "${target_dataset}" \
                --epochs ${FT_EPOCHS} \
                --lr ${FT_LR} \
                --patience ${FT_PATIENCE} \
                --bs ${FT_BATCH_SIZE} \
                --freeze-mode ${FT_FREEZE_MODE}
        fi

        # Testing & Aggregation Phase
        echo -e "\n--- Testing and Aggregating Results for ${RUN_ID} ---"
        
        RESULTS_DIR="results"
        mkdir -p ${RESULTS_DIR}
        RESULTS_CSV="${RESULTS_DIR}/results_${RUN_ID}.csv"
        echo "PA,DSC,HD95,IoU,GFLOPs,Params" > ${RESULTS_CSV}

        for fold in $(seq 1 ${NUM_FOLDS}); do
            FT_CHECKPOINT_PATH="${CHECKPOINT_BASE_DIR}/${RUN_ID}_fold${fold}_best.pth"
            
            if [ -f "$FT_CHECKPOINT_PATH" ]; then
                TEST_OUTPUT=$(python test.py -r "$FT_CHECKPOINT_PATH")
                
                PA=$(echo "$TEST_OUTPUT" | grep "PA:" | cut -d':' -f2 | xargs)
                DSC=$(echo "$TEST_OUTPUT" | grep "DSC:" | cut -d':' -f2 | xargs)
                HD95=$(echo "$TEST_OUTPUT" | grep "HD95:" | cut -d':' -f2 | xargs)
                IOU=$(echo "$TEST_OUTPUT" | grep "IoU:" | cut -d':' -f2 | xargs)
                GFLOPS=$(echo "$TEST_OUTPUT" | grep "GFLOPs:" | cut -d':' -f2 | xargs)
                PARAMS=$(echo "$TEST_OUTPUT" | grep "Params:" | cut -d':' -f2 | xargs)
                
                echo "${PA},${DSC},${HD95},${IOU},${GFLOPS},${PARAMS}" >> ${RESULTS_CSV}
            else
                echo "Error: Checkpoint for fold ${fold} not found at ${FT_CHECKPOINT_PATH}!"
            fi
        done
        
        echo "--- EXPERIMENT FINISHED: ${RUN_ID} ---"
    done
done

echo -e "\n\n======================================================="
echo "  ALL TRANSFER LEARNING EXPERIMENTS COMPLETED"
echo "======================================================="