#!/bin/bash
#BCM Training
python main.py train --config-path configs/voc12_bcm.yaml

#FR-Shifting Training
python main.py train --config-path configs/voc12_fr.yaml

#Testing
python main.py test --config-path configs/voc12_fr.yaml --model-path data/models/voc12/FR/checkpoint_final.pth

#Generating Pseudo Labels for Semantic Segmenation
python main.py test --config-path configs/voc12_fr.yaml --model-path data/models/voc12/FR/checkpoint_final.pth --gen-training True
python main.py gen --config-path configs/voc12_fr.yaml

#Generating Pseudo Labels for Instance Segmenation
python make_coco_inst_mask_label.py
