#!/bin/sh
cd ..

python BMNet_Run.py --split val --gpu_id 4 --dataset fscdlvis
python BMNet_Run.py --split test --gpu_id 4 --dataset fscdlvis