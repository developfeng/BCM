#!/bin/bash
#BCM Training
python main.py train --config-path configs/voc12_bcm.yaml

#FR-Shifting Training
python main.py train --config-path configs/voc12_fr.yaml

#Generating Pseudo Labels for Semantic Segmenation
python main.py gen --config-path configs/voc12_fr.yaml --model-path data/models/voc12/deeplabv2_resnet101_msc_bcm/trainaug/checkpoint_final.pth

#Generating Pseudo Labels for Instance Segmenation
python main.py crf --config-path configs/voc12_crf.yaml