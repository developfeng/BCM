# BCM and FR-Loss
--------------------------------------------------------------------------------
* Weakly Supervised Semantic Segmentation via Box-driven Masking and Filling Rate Shifting 
* Code Version 1.0                                                             
* E-mail: chunfeng.song@nlpr.ia.ac.cn                                          
---------------------------------------------------------------------------------

i.    Overview
ii.   Copying
iii.  Use

i. OVERVIEW
-----------------------------
This code implements our paper:

>Weakly Supervised Semantic Segmentation via Box-driven Masking and Filling Rate Shifting.

and reimplement the Conference Version with PyTorch:

>Box-driven Class-wise Region Masking and Filling Rate Guided Loss for Weakly Supervised Semantic Segmentation, CVPR, 2019.


If you find this work is helpful for your research, please cite our paper [[PDF]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Song_Box-Driven_Class-Wise_Region_Masking_and_Filling_Rate_Guided_Loss_for_CVPR_2019_paper.pdf).

ii. COPYING
-----------------------------
We share this code only for research use. We neither warrant 
correctness nor take any responsibility for the consequences of 
using this code. If you find any problem or inappropriate content
in this code, feel free to contact us (chunfeng.song@nlpr.ia.ac.cn).

iii. USE
-----------------------------
This code should work on PyTorch and based on the widely used [DeepLabV2](https://github.com/kazuto1011/deeplab-pytorch) implementation. 

(1) Data Preparation.

Download the VOC2012 dataset following the guideline from [DeepLabV2](https://github.com/kazuto1011/deeplab-pytorch) and the init pseudo seeds M&G+ from [SDS](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/weakly-supervised-learning/simple-does-it-weakly-supervised-instance-and-semantic-segmentation).
 Our pretrained models and the final generated pseudo labels can be download from [GoogleDrive](https://drive.google.com/drive/folders/1BluuWCms0LLCW6zcNlE8hsvTJIY5n32o?usp=sharing).
 
(2) Model Training.

BCM Training
```bash
python main.py train --config-path configs/voc12_bcm.yaml
```

FR-Shifting Training
```bash
python main.py train --config-path configs/voc12_fr.yaml
```

Testing
```bash
python main.py test --config-path configs/voc12_fr.yaml --model-path data/models/voc12/FR/checkpoint_final.pth
```

Generating Pseudo Labels for Semantic Segmenation
```bash
python main.py test --config-path configs/voc12_fr.yaml --model-path data/models/voc12/FR/checkpoint_final.pth --gen-training True
```
```bash
python main.py gen --config-path configs/voc12_fr.yaml
```

Generating Pseudo Labels for Instance Segmenation
```bash
python make_coco_inst_mask_label.py
```

## Related links
 * DeepLabV2: https://github.com/kazuto1011/deeplab-pytorch
