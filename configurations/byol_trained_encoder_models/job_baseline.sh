#!/bin/bash

python main_tram.py -c "scripts/byol_trained_encoder_models/finetune.json" --cl 1 --n BYOL_Baselines
python main_tram.py -c "scripts/byol_trained_encoder_models/linear.json" --cl 1 --n BYOL_Baselines

python main_tram.py -c "scripts/byol_trained_encoder_models/byol_tram.json" --en separate --fc 1 --cl 1 --n BYOL_Baselines
python main_tram.py -c "scripts/byol_trained_encoder_models/byol_tram.json" --en separate --fc 0 --cl 1 --n BYOL_Baselines

python main_tram.py -c "scripts/byol_trained_encoder_models/byol_tram.json" --en combined --fc 1 --cl 1 --n BYOL_Baselines
python main_tram.py -c "scripts/byol_trained_encoder_models/byol_tram.json" --en combined --fc 0 --cl 1 --n BYOL_Baselines

python main_tram.py -c "scripts/byol_trained_encoder_models/byol_tram.json" --en onehot --fc 1 --cl 1 --n BYOL_Baselines
python main_tram.py -c "scripts/byol_trained_encoder_models/byol_tram.json" --en onehot --fc 0 --cl 1 --n BYOL_Baselines


