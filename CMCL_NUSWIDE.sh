#!/bin/bash

################# bash -x ***.sh #################
lr=0.002
gpu_rank=0
valid_freq=2
epochs=10
res_name="Result_CMCL_NUSWIDE"
recon=0.001

python main.py --is-train --dataset nuswide --query-num 2100 --train-num 10500 --lr "$lr" --rank "$gpu_rank" --valid-freq "$valid_freq" --epochs "$epochs" --result-name "$res_name" --hyper-recon "$recon"
