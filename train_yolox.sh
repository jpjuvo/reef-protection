#!/bin/bash
. configs/yolox.config

# Printouts
echo IMG_SZ "$IMG_SZ"
echo EPOCHS "$EPOCHS"
echo BATCH_SIZE "$BATCH_SIZE"
echo WEIGHTS "$WEIGHTS"

cd YOLOX

for fold in 0 #0 1 2 3 4 
do
  echo Training fold "$fold"
  NAME=SAMPLE5_"$IMG_SZ"_fold"$fold"_"$MODEL"-"$EPOCHS"ep
  echo NAME "$NAME"

  python ../src/train_yolox.py \
    --img $IMG_SZ \
    --epochs $EPOCHS \
    --fold $fold

  python ./tools/train.py \
    -f "../configs/yolox_last_config.py" \
    -d 1 \
    -b $BATCH_SIZE \
    --fp16 \
    -o \
    -c $WEIGHTS



    --data ../input/yolo_ds/fold_"$fold".yaml \
    --hyp ../configs/"$HYPERPARAMS" \
    --adam \
    --cfg models/"$MODEL".yaml \
    --weights $WEIGHTS \
    --name "$NAME" \
    --entity $ENTITY \
    --project $PROJECT
done