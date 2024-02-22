#!/bin/bash

################# bash -x ***.sh #################
lr=0.002
gpu_rank=1
valid_freq=3
epochs=30
res_name="Result_CMCL_COCO"
recon=0.005

python main.py --is-train  --dataset coco --query-num 5000 --train-num 10000 --lr "$lr" --rank "$gpu_rank" --valid-freq "$valid_freq" --epochs "$epochs" --result-name "$res_name" --hyper-recon "$recon"
