#!/bin/bash
. configs/yolo_base.config

NAME=3000_fold3_"$MODEL"-"$EPOCHS"ep

# Printouts
echo NAME "$NAME"
echo IMG_SZ "$IMG_SZ"
echo EPOCHS "$EPOCHS"
echo BATCH_SIZE "$BATCH_SIZE"
echo WEIGHTS "$WEIGHTS"
echo HYPERPARAMS "$HYPERPARAMS"

cd yolov5

for fold in 3 #0 1 2 3 4 
do
  echo Training fold "$fold"
  
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
    --name "$NAME""$fold" \
    --entity $ENTITY \
    --project $PROJECT
done