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

Download the init seeds data and their masks from:

(2) Model Training.

#BCM Training

python main.py train --config-path configs/voc12_bcm.yaml

#FR-Shifting Training

python main.py train --config-path configs/voc12_fr.yaml

#Generating Pseudo Labels for Semantic Segmenation

python main.py gen --config-path configs/voc12_fr.yaml --model-path data/models/voc12/deeplabv2_resnet101_msc_bcm/trainaug/checkpoint_final.pth

#Generating Pseudo Labels for Instance Segmenation
