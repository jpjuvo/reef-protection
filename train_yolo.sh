#!/bin/bash
. configs/yolo_base.config

NAME=AUG2_"$MODEL"-"$EPOCHS"ep

# Printouts
echo NAME "$NAME"
echo IMG_SZ "$IMG_SZ"
echo EPOCHS "$EPOCHS"
echo BATCH_SIZE "$BATCH_SIZE"
echo WEIGHTS "$WEIGHTS"
echo HYPERPARAMS "$HYPERPARAMS"

cd yolov5

for fold in 0 #1 2 3 4 
do
  echo Training fold "$fold"
  python train.py \
    --img $IMG_SZ \
    --batch $BATCH_SIZE \
    --epochs $EPOCHS \
    --data ../input/yolo_ds/fold_"$fold".yaml \
    --hyp ../configs/"$HYPERPARAMS" \
    --adam \
    --cfg models/hub/"$MODEL".yaml \
    --weights $WEIGHTS \
    --name "$NAME""$fold" \
    --entity $ENTITY \
    --project $PROJECT
done