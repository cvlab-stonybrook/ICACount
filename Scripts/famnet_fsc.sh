#!/bin/sh
cd ..

python FamNet_Run.py --split test --gpu_id 4 --dataset fsc147
python FamNet_Run.py --split val --gpu_id 4 --dataset fsc147
