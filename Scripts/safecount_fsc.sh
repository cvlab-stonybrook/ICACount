#!/bin/sh
cd ..

python SAFECount_Run.py --split val --gpu_id 5 --dataset fsc147
python SAFECount_Run.py --split test --gpu_id 5 --dataset fsc147