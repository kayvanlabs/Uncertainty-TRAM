#!/bin/bash

python main_tram.py -c "scripts/supervised_trained_encoder_models/proposed.json" --en onehot --fc 1 --th 0.5 --cl 1 --n Supervised_Thred_Check
python main_tram.py -c "scripts/supervised_trained_encoder_models/proposed.json" --en onehot --fc 1 --th 1.0 --cl 1 --n Supervised_Thred_Check
python main_tram.py -c "scripts/supervised_trained_encoder_models/proposed.json" --en onehot --fc 1 --th 1.5 --cl 1 --n Supervised_Thred_Check
python main_tram.py -c "scripts/supervised_trained_encoder_models/proposed.json" --en onehot --fc 1 --th 2.0 --cl 1 --n Supervised_Thred_Check
python main_tram.py -c "scripts/supervised_trained_encoder_models/proposed.json" --en onehot --fc 1 --th 2.5 --cl 1 --n Supervised_Thred_Check
python main_tram.py -c "scripts/supervised_trained_encoder_models/proposed.json" --en onehot --fc 1 --th 3.0 --cl 1 --n Supervised_Thred_Check
python main_tram.py -c "scripts/supervised_trained_encoder_models/proposed.json" --en onehot --fc 1 --th 3.5 --cl 1 --n Supervised_Thred_Check
python main_tram.py -c "scripts/supervised_trained_encoder_models/proposed.json" --en onehot --fc 1 --th 4.0 --cl 1 --n Supervised_Thred_Check