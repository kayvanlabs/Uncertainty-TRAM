#!/bin/bash

python main_tram.py -c "scripts/supervised_trained_encoder_models/proposed.json" --cl 1 --en onehot --fc 1 --n Supervised_Proposed_Models
python main_tram.py -c "scripts/supervised_trained_encoder_models/proposed.json" --cl 1 --en onehot --fc 0 --n Supervised_Proposed_Models

python main_tram.py -c "scripts/supervised_trained_encoder_models/proposed.json" --cl 1 --en separate --fc 1 --n Supervised_Proposed_Models
python main_tram.py -c "scripts/supervised_trained_encoder_models/proposed.json" --cl 1 --en separate --fc 0 --n Supervised_Proposed_Models

python main_tram.py -c "scripts/supervised_trained_encoder_models/proposed.json" --cl 1 --en combined --fc 1 --n Supervised_Proposed_Models
python main_tram.py -c "scripts/supervised_trained_encoder_models/proposed.json" --cl 1 --en combined --fc 0 --n Supervised_Proposed_Models