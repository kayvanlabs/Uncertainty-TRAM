#!/bin/bash

python main_tram.py -c "scripts/confusion_matrix_baseline/finetune_confusion_transformer.json" --cl 1 --n Confusion_Estimation
python main_tram.py -c "scripts/confusion_matrix_baseline/finetune_confusion_supervised.json" --cl 1 --n Confusion_Estimation
python main_tram.py -c "scripts/confusion_matrix_baseline/finetune_confusion_byol.json" --cl 1 --n Confusion_Estimation