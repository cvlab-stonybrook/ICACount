#!/bin/sh
cd ..

python FamNet_Run.py --split test --gpu_id 3 --dataset fscdlvis
python FamNet_Run.py --split val --gpu_id 3 --dataset fscdlvis
