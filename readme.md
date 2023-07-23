## Leveraging Uncertainty as Privileged Information for Acute Respiratory Distress Syndrome Detection in Chest X-ray Images


### Requirements

The project requires the following Python packages and can be easily installed by `pip install -r requirements.txt`.

```txt
coral_pytorch==1.4.0
kornia==0.6.10
numpy==1.21.5
opencv_python==4.7.0.68
pandas==1.3.5
scikit_image==0.19.2
scikit_learn==1.0.2
torch==1.13.1
torchmetrics==0.11.1
torchvision==0.14.1
torchxrayvision==0.0.39
```

### Experiments

#### Experiments on Supervised Pretrained Encoder

1. Baseline v.s. Proposed

```bash
bash configurations\supervised_trained_encoder_models\job_baseline.sh
bash configurations\supervised_trained_encoder_models\job_proposed.sh
```

2. Explore the Thresholding

```bash
bash configurations\supervised_trained_encoder_models\job_thred_check.sh
```
#### Confusion Estimation Baselines

```bash
bash configurations\confusion_matrix_baseline\job.sh
```

#### BYOL Pretraining of Encoder

```bash
bash configurations/byol_train/train.sh
```

#### Experiments on BYOL Pretrained Encoder 

```bash
bash configurations\byol_trained_encoder_models\job_baseline.sh
bash configurations\byol_trained_encoder_models\job_proposed.sh
```

#### Experiments on DINO Pretrained Encoder 

```bash
bash configurations\vit_encoder_models\job_baseline.sh
bash configurations\vit_encoder_models\job_proposed.sh
```

### Module Structure

#### For Running `main.py`

```txt
 | --- main.py
     | --- data_loader.py
            | --- dataset.py
     | --- models.py
     | --- parse_config.py
     | --- main_utils.py
            | --- customized_scheduler.py
     | --- logger_utils.py
            | --- main_utils.py
            | --- logger_config.json
     | --- augmentations.py
     | --- confusion_utils.py
     | --- trainer_trams.py
            | --- metric.py
            | --- logger_utils.py
            | --- loss_trams.py
            | --- confusion_utils.py
```
#### For BYOL pretraining main file `main_ssl.py`

```txt
 | --- main_ssl.py
     | --- data_loader.py
            | --- dataset.py
     | --- models.py
     | --- parse_config.py
     | --- main_utils.py
            | --- customized_scheduler.py
     | --- logger_utils.py
            | --- main_utils.py
            | --- logger_config.json
     | --- augmentations.py
     | --- confusion_utils.py
     | --- trainer_trams.py
            | --- metric.py
            | --- logger_utils.py
            | --- loss_trams.py
            | --- confusion_utils.py
```

If you have any questions, please contact us via: zijung@umich.edu