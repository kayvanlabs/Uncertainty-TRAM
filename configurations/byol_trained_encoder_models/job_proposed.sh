#!/bin/bash

python main_tram.py -c "scripts/byol_trained_encoder_models/proposed.json" --en separate --fc 1 --cl 1 --n BYOL_Proposed_Models
python main_tram.py -c "scripts/byol_trained_encoder_models/proposed.json" --en separate --fc 0 --cl 1 --n BYOL_Proposed_Models

python main_tram.py -c "scripts/byol_trained_encoder_models/proposed.json" --en combined --fc 1 --cl 1 --n BYOL_Proposed_Models
python main_tram.py -c "scripts/byol_trained_encoder_models/proposed.json" --en combined --fc 0 --cl 1 --n BYOL_Proposed_Models

python main_tram.py -c "scripts/byol_trained_encoder_models/proposed.json" --en onehot --fc 1 --cl 1 --n BYOL_Proposed_Models
python main_tram.py -c "scripts/byol_trained_encoder_models/proposed.json" --en onehot --fc 0 --cl 1 --n BYOL_Proposed_Models
