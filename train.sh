#!/bin/bash

set -eu

train_steps=2500
dev_steps=800
chunk_size="140,180"
cpt_dir=exp/ge2e
epochs=50

echo "$0 $@"

[ $# -ne 2 ] && echo "Script format error: $0 <gpuid> <cpt-id>" && exit 1

# DEBUG
./ge2e/train_ge2e.py --M 2 --N 2 --gpu 0 --epochs 1 --train-steps 100 --dev-steps 80 --chunk-size "140,180" --checkpoint ../exp/ge2e/1

./ge2e/train_ge2e.py \
  --M 10 \
  --N 64 \
  --gpu $1 \
  --epochs $epochs \
  --train-steps $train_steps \
  --dev-steps $dev_steps \
  --chunk-size $chunk_size \
  --checkpoint $cpt_dir/$2 \
  > $2.train.log 2>&1 