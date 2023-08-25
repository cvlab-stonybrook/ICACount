#!/bin/sh
cd ..

python FamNet_Run.py --split val --gpu_id 2 --dataset fsc147
python FamNet_Run.py --split test --gpu_id 2 --dataset fsc147