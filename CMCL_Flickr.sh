#!/bin/bash

################# bash -x ***.sh #################
lr=0.001
gpu_rank=2
valid_freq=3
epochs=30
res_name="Result_CMCL_Flickr"
recon=0.001

python main.py --is-train --dataset flickr25k --query-num 2000 --train-num 10000  --lr "$lr" --rank "$gpu_rank" --valid-freq "$valid_freq" --epochs "$epochs" --result-name "$res_name" --hyper-recon "$recon"
