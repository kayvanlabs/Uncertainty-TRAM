#!/bin/bash

python main_tram.py -c "scripts/vit_encoder_models/finetune.json" --cl 1 --n Vit_Baseline
python main_tram.py -c "scripts/vit_encoder_models/probe.json" --cl 1 --n Vit_Baseline

python main_tram.py -c "scripts/vit_encoder_models/tram.json" --en separate --fc 1 --cl 1 --n Vit_Baseline
python main_tram.py -c "scripts/vit_encoder_models/tram.json" --en separate --fc 0 --cl 1 --n Vit_Baseline

python main_tram.py -c "scripts/vit_encoder_models/tram.json" --en onehot --fc 1 --cl 1 --n Vit_Baseline
python main_tram.py -c "scripts/vit_encoder_models/tram.json" --en onehot --fc 0 --cl 1 --n Vit_Baseline

python main_tram.py -c "scripts/vit_encoder_models/tram.json" --en combined --fc 1 --cl 1 --n Vit_Baseline
python main_tram.py -c "scripts/vit_encoder_models/tram.json" --en combined --fc 0 --cl 1 --n Vit_Baseline
