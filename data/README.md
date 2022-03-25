## Pretrained models and data

(1) init_pretrained_model: Model pretrained with default settings of [DeeplabV2](https://github.com/kazuto1011/deeplab-pytorch) on the seeds mask of M&G+ from [SDS](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/weakly-supervised-learning/simple-does-it-weakly-supervised-instance-and-semantic-segmentation).

(2) bcm_model: Model finetuned with BCM supervision.

(3) fr_model: Final model finetuned under FR loss.

(4) pseudo_semantic_labels_trainaug: the final generated pseudo labels of train_aug set, which can be used for further training with any fully-supervised models.

(5) pseudo_inst_labels_trainaug_val: the COCO-like instance pseudo labels of train_aug and val set, which can be used for further training with instance segmentation models like Mask RCNN in [Detectron2](https://github.com/facebookresearch/detectron2) and [MMDetction](https://github.com/open-mmlab/mmdetection).
