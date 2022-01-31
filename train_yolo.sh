#!/bin/bash
. configs/yolo_base.config

# Printouts
echo IMG_SZ "$IMG_SZ"
echo EPOCHS "$EPOCHS"
echo BATCH_SIZE "$BATCH_SIZE"
echo WEIGHTS "$WEIGHTS"
echo HYPERPARAMS "$HYPERPARAMS"

cd yolov5

for fold in 0 #0 1 2 3 4 
do
  echo Training fold "$fold"
  NAME=SAMPLE5_"$IMG_SZ"_fold"$fold"_"$MODEL"-"$EPOCHS"ep
  echo NAME "$NAME"
  
  # yolo5x6 models are in /hub subdir
  # --cfg models/hub/"$MODEL".yaml \
  # yolo5 models on models dir
  # --cfg models/"$MODEL".yaml \

  python train.py \
    --img $IMG_SZ \
    --batch $BATCH_SIZE \
    --epochs $EPOCHS \
    --data ../input/yolo_ds/fold_"$fold".yaml \
    --hyp ../configs/"$HYPERPARAMS" \
    --adam \
    --cfg models/"$MODEL".yaml \
    --weights $WEIGHTS \
    --name "$NAME" \
    --entity $ENTITY \
    --project $PROJECT
done