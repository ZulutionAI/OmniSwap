#!/bin/bash

cd ../../.. || exit
SAPIENS_CHECKPOINT_ROOT=checkpoints/sapiens/sapiens_host

#----------------------------set your input and output directories----------------------------------------------
INPUT="xxx/img"  # must end with /img
OUTPUT="xxx/mask_sapiens"

mkdir $OUTPUT
#--------------------------MODEL CARD---------------
# MODEL_NAME='sapiens_0.3b'; CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/seg/checkpoints/sapiens_0.3b/sapiens_0.3b_goliath_best_goliath_mIoU_7673_epoch_194.pth
# MODEL_NAME='sapiens_0.6b'; CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/seg/checkpoints/sapiens_0.6b/sapiens_0.6b_goliath_best_goliath_mIoU_7777_epoch_178.pth
MODEL_NAME='sapiens_1b'; CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/seg/checkpoints/sapiens_1b/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151.pth
# MODEL_NAME='sapiens_2b'; CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/seg/checkpoints/sapiens_2b/sapiens_2b_goliath_best_goliath_mIoU_8131_epoch_200.pth

DATASET='goliath'
MODEL="${MODEL_NAME}_${DATASET}-1024x768"
CONFIG_FILE="data_process/sapiens/seg/sapiens_seg/${DATASET}/${MODEL}.py"
# OUTPUT=$OUTPUT/$MODEL_NAME

##-------------------------------------inference-------------------------------------
RUN_FILE='data_process/sapiens/seg/demo_seg_vis.py'

## number of inference jobs per gpu, total number of gpus and gpu ids
JOBS_PER_GPU=1; TOTAL_GPUS=8; VALID_GPU_IDS=(0 1 2 3 4 5 6 7)  # (0 1 2 3 4 5 6 7)
TOTAL_JOBS=$((JOBS_PER_GPU * TOTAL_GPUS))

# Find all images and sort them, then write to a temporary text file
IMAGE_LIST="${OUTPUT}/img_list.txt"
python ../collect_data.py --input $INPUT --output_root $OUTPUT --process_type mask || exit 1

# Check if image list was created successfully
if [ ! -s "${IMAGE_LIST}" ]; then
  echo "No images found. Check your input directory and permissions."
  exit 1
fi

# Count images and calculate the number of images per text file
NUM_IMAGES=$(wc -l < "${IMAGE_LIST}")
IMAGES_PER_FILE=$((NUM_IMAGES / TOTAL_JOBS))
EXTRA_IMAGES=$((NUM_IMAGES % TOTAL_JOBS))

export TF_CPP_MIN_LOG_LEVEL=2
echo "Distributing ${NUM_IMAGES} image paths into ${TOTAL_JOBS} jobs."

# Divide image paths into text files for each job
for ((i=0; i<TOTAL_JOBS; i++)); do
  TEXT_FILE="${OUTPUT}/image_paths_$((i+1)).txt"
  if [ $i -eq $((TOTAL_JOBS - 1)) ]; then
    # For the last text file, write all remaining image paths
    tail -n +$((IMAGES_PER_FILE * i + 1)) "${IMAGE_LIST}" > "${TEXT_FILE}"
  else
    # Write the exact number of image paths per text file
    head -n $((IMAGES_PER_FILE * (i + 1))) "${IMAGE_LIST}" | tail -n ${IMAGES_PER_FILE} > "${TEXT_FILE}"
  fi
done

# Run the process on the GPUs, allowing multiple jobs per GPU
for ((i=0; i<TOTAL_JOBS; i++)); do
  GPU_ID=$((i % TOTAL_GPUS))
  CUDA_VISIBLE_DEVICES=${VALID_GPU_IDS[GPU_ID]} python ${RUN_FILE} \
    ${CONFIG_FILE} \
    ${CHECKPOINT} \
    --input "${OUTPUT}/image_paths_$((i+1)).txt" \
    --output-root="${OUTPUT}" & ## add & to process in background
  # Allow a short delay between starting each job to reduce system load spikes
  sleep 1
  echo "Process in GPU ${VALID_GPU_IDS[GPU_ID]} start!!!!!!!!!"
done

# Wait for all background processes to finish
wait

# Remove the image list and temporary text files
# rm "${IMAGE_LIST}"
for ((i=0; i<TOTAL_JOBS; i++)); do
  rm "${OUTPUT}/image_paths_$((i+1)).txt"
done

# Go back to the original script's directory
cd -

echo "Processing complete."
echo "Results saved to $OUTPUT"


# 0: Background
# 1: Apparel
# 2: Face_Neck
# 3: Hair
# 4: Left_Foot
# 5: Left_Hand
# 6: Left_Lower_Arm
# 7: Left_Lower_Leg
# 8: Left_Shoe
# 9: Left_Sock
# 10: Left_Upper_Arm
# 11: Left_Upper_Leg
# 12: Lower_Clothing
# 13: Right_Foot
# 14: Right_Hand
# 15: Right_Lower_Arm
# 16: Right_Lower_Leg
# 17: Right_Shoe
# 18: Right_Sock
# 19: Right_Upper_Arm
# 20: Right_Upper_Leg
# 21: Torso
# 22: Upper_Clothing
# 23: Lower_Lip
# 24: Upper_Lip
# 25: Lower_Teeth
# 26: Upper_Teeth
# 27: Tongue