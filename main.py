#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import json
import multiprocessing
import os
import datetime
from time import strftime, localtime
import click
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchnet.meter import MovingAverageValueMeter
from tqdm import tqdm

from libs.datasets import get_dataset
from libs.models import DeepLabV2_ResNet101_MSC_BCM
from libs.utils import DenseCRF, PolynomialLR, scores
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'#Edit to fit the GPU number.

def makedirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)


def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        print("Device:")
        for i in range(torch.cuda.device_count()):
            print("    {}:".format(i), torch.cuda.get_device_name(i))
    else:
        print("Device: CPU")
    return device


def get_params(model, key):
    # For Dilated FCN
    if key == "1x":
        for m in model.named_modules():
            if "layer" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    for p in m[1].parameters():
                        yield p
    # For conv weight in the ASPP module
    if key == "10x":
        for m in model.named_modules():
            if "aspp" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    yield m[1].weight
    # For conv bias in the ASPP module
    if key == "20x":
        for m in model.named_modules():
            if "aspp" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    yield m[1].bias

def make_FR_labels(labels, boxes, logits, bcm, size):
    """
    Producing the Filling-Rate labels, with anchor-based shifting.
    """
    fr_ratio_per_class = [0.3460136720164967, 0.5389285952169538, 0.5588745925915763, 0.5450067976049356, 0.5333692537505409, 0.5910699748072926, 0.6545625904725838, 0.6233635810717549, 0.637256951605831, 0.49807774097729646, 0.6643279384114361, 0.5056760126964205, 0.5917462320307347, 0.6116644853677702, 0.6084525573744772, 0.5942225745380297, 0.5924586736754247, 0.6659348466060375, 0.4216027245277032, 0.5823734251549558]
    new_labels = []
    probs = F.softmax(logits, dim=1)
    for i in range(labels.shape[0]):
        label = labels[i]
        box = boxes[i]
        label = label.float().numpy()
        box = box.float().numpy()
        label = Image.fromarray(label).resize(size, resample=Image.NEAREST)
        box = Image.fromarray(box).resize(size, resample=Image.NEAREST)
        box = np.asarray(box)
        label_copy = np.zeros(size, dtype = np.float)
        label = np.asarray(label)
        label_copy[:] = label[:]
        total_ignore = 0
        for cls in range(probs.shape[1]):
            box_pixel = np.sum(box==cls)
            label_pixel = np.sum(label==cls)
            if label_pixel>box_pixel:
                label_pixel = box_pixel #The min number of forground pixels. 
            
            box_mask = (box == cls) #Select one class
            pseudo_mask = (label == cls)
            if box_pixel<=2 or cls==0:
                if cls==0:
                    label_copy[box == 0]=0 #The 'background' lables in box is accurate.
                else:
                    continue #Skip if no pixels belong to this class.
            else:
                valid_mask = (box == cls) #Valid mask only works within this class box.
                fr_ratio = fr_ratio_per_class[cls-1] #There is no 'background' filling rate, so start from the first object class.
                cls_pixel = int(box_pixel*fr_ratio) #Anchor filling rate, calculating with the default class-wise fr-ratio.
                this_prob = probs[i,cls,:,:].cpu().detach().numpy()
                sort_prob = np.sort(-this_prob[valid_mask])
                
                # Evaluate the variances of the divided groups.
                sigma_a = np.var(sort_prob[:cls_pixel])
                sigma_b = np.var(sort_prob[(cls_pixel-1):])
                delta = (sigma_b-sigma_a)/max(sigma_a,sigma_b) #The shifting factor.
                if delta>1:
                    delta = 0
                fr_ratio = fr_ratio*(1+0.3*delta) #Adjust the anchor FR with the shifting factor, in which 0.3 is a tunned weights.
                
                # Normalize the value range.
                cls_pixel = min(max(label_pixel,int(box_pixel*fr_ratio)), box_pixel-1)
                this_threshold = min(sort_prob[cls_pixel-1], 0.05)
                this_threshold_up = max(sort_prob[int(cls_pixel)],0.98)
                
                #Select the gray zone by the FR-Shifting.
                region_to_ignore = (this_prob[box_mask] <= this_threshold) & (label[box_mask]==cls)
                valid_mask[box_mask] = region_to_ignore
                label_copy[valid_mask]=255 #Ignore them in training.
                
                # The wrongly assigned lables (that however with high confidence score) should be adjusted.
                region_to_change = (this_prob[box_mask] > this_threshold_up) & (label[box_mask]==255)
                label_copy[box_mask][region_to_change]=cls
                
        if (sum(sum(label_copy!=label))/(size[0]*size[1]))>0.05: #Only effect with sufficient pixels.
            new_labels.append(np.asarray(label))
        else:
            new_labels.append(np.asarray(label_copy))
    new_labels = torch.LongTensor(new_labels)
    return new_labels

def resize_labels(labels, size, to_one_hot=False):
    """
    Downsample labels for 0.5x and 0.75x logits by nearest interpolation.
    Other nearest methods result in misaligned labels.
    -> F.interpolate(labels, shape, mode='nearest')
    -> cv2.resize(labels, shape, interpolation=cv2.INTER_NEAREST)
    to_one_hot: for BCM supervision producing.    
    """
    new_labels = []
    for label in labels:
        label = label.float().numpy()
        label = Image.fromarray(label).resize(size[-2:], resample=Image.NEAREST)
        if to_one_hot: #Producing the BCM supervision.
            box_label = np.zeros(size, dtype = np.float)
            label = np.asarray(label)
            ty, tx = (label == 255).nonzero()
            box_label +=0.5 #Soft bcm with default values from zero to 0.5.
            box_label[:,ty,tx] = 1
            for cls in range(size[0]):
                ty, tx = (label == cls).nonzero()
                box_label[cls,ty,tx] = 1
            box_label[0,:,:] = 1 #All backgroung are useful with highest weights.
            new_labels.append(np.asarray(box_label))
        else:        
            new_labels.append(np.asarray(label))
    if to_one_hot:
        new_labels = torch.FloatTensor(new_labels)
    else:
        new_labels = torch.LongTensor(new_labels)
    return new_labels


@click.group()
@click.pass_context
def main(ctx):
    """
    Training and evaluation
    """
    print("Mode:", ctx.invoked_subcommand)


@main.command()
@click.option(
    "-c",
    "--config-path",
    type=click.File(),
    required=True,
    help="Dataset configuration file in YAML",
)
@click.option(
    "--cuda/--cpu", default=True, help="Enable CUDA if available [default: --cuda]"
)

def train(config_path, cuda):
    """
    Training DeepLab by v2 protocol
    """

    # Configuration
    CONFIG = OmegaConf.load(config_path)
    device = get_device(cuda)
    torch.backends.cudnn.benchmark = True

    # Dataset
    dataset = get_dataset(CONFIG.DATASET.NAME)(
        root=CONFIG.DATASET.ROOT,
        split=CONFIG.DATASET.SPLIT.TRAIN,
        ignore_label=CONFIG.DATASET.IGNORE_LABEL,
        mean_bgr=(CONFIG.IMAGE.MEAN.B, CONFIG.IMAGE.MEAN.G, CONFIG.IMAGE.MEAN.R),
        augment=True,
        base_size=CONFIG.IMAGE.SIZE.BASE,
        crop_size=CONFIG.IMAGE.SIZE.TRAIN,
        scales=CONFIG.DATASET.SCALES,
        flip=True,
    )
    print(dataset)

    # DataLoader
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=CONFIG.SOLVER.BATCH_SIZE.TRAIN,
        num_workers=CONFIG.DATALOADER.NUM_WORKERS,
        shuffle=True,
    )
    loader_iter = iter(loader)

    # Model check
    print("Model:", CONFIG.MODEL.NAME)
    assert (
        CONFIG.MODEL.NAME == "DeepLabV2_ResNet101_MSC_BCM"
    ), 'Currently support only "DeepLabV2_ResNet101_MSC" and "DeepLabV2_ResNet101_MSC_BCM".'

    # Model setup
    model = DeepLabV2_ResNet101_MSC_BCM(n_classes=CONFIG.DATASET.N_CLASSES)
    state_dict = torch.load(CONFIG.MODEL.INIT_MODEL)
    print("    Init:", CONFIG.MODEL.INIT_MODEL)
    '''
    for m in model.base.state_dict().keys():
        if m not in state_dict.keys():
            print("    Skip init:", m)
    model.base.load_state_dict(state_dict, strict=False)  # to skip ASPP
    '''
    model.load_state_dict(state_dict, strict=False)#Finetune from base model.
    model = nn.DataParallel(model)
    model.to(device)

    # Loss definition
    CEL = nn.CrossEntropyLoss(ignore_index=CONFIG.DATASET.IGNORE_LABEL)
    CEL.to(device)
    MSEL = nn.MSELoss().to(device)
    FR_Shifting = CONFIG.MODEL.FR_ON

    # Optimizer
    optimizer = torch.optim.SGD(
        # cf lr_mult and decay_mult in train.prototxt
        params=[
            {
                "params": get_params(model.module, key="1x"),
                "lr": CONFIG.SOLVER.LR,
                "weight_decay": CONFIG.SOLVER.WEIGHT_DECAY,
            },
            {
                "params": get_params(model.module, key="10x"),
                "lr": 10 * CONFIG.SOLVER.LR,
                "weight_decay": CONFIG.SOLVER.WEIGHT_DECAY,
            },
            {
                "params": get_params(model.module, key="20x"),
                "lr": 20 * CONFIG.SOLVER.LR,
                "weight_decay": 0.0,
            },
        ],
        momentum=CONFIG.SOLVER.MOMENTUM,
    )

    # Learning rate scheduler
    scheduler = PolynomialLR(
        optimizer=optimizer,
        step_size=CONFIG.SOLVER.LR_DECAY,
        iter_max=CONFIG.SOLVER.ITER_MAX,
        power=CONFIG.SOLVER.POLY_POWER,
    )

    # Setup loss logger
    writer = SummaryWriter(os.path.join(CONFIG.EXP.OUTPUT_DIR, "logs", CONFIG.EXP.ID))
    average_loss = MovingAverageValueMeter(CONFIG.SOLVER.AVERAGE_LOSS)

    # Path to save models
    checkpoint_dir = os.path.join(
        CONFIG.EXP.OUTPUT_DIR,
        "models",
        CONFIG.EXP.ID,
        CONFIG.EXP.NAME,
    )
    log_path = checkpoint_dir + '/log_%s.txt'%datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    makedirs(checkpoint_dir)
    print("Checkpoint dst:", checkpoint_dir)
    log_file = open(log_path, 'w')
    log_file.write("Checkpoint dst:%s"%checkpoint_dir + '\n')
    log_file.flush()
    # Freeze the batch norm pre-trained on COCO
    model.train()
    model.module.base.freeze_bn()

    for iteration in tqdm(
        range(1, CONFIG.SOLVER.ITER_MAX + 1),
        total=CONFIG.SOLVER.ITER_MAX,
        dynamic_ncols=True,
    ):

        # Clear gradients (ready to accumulate)
        optimizer.zero_grad()

        loss = 0
        for _ in range(CONFIG.SOLVER.ITER_SIZE):
            try:
                image_ids, images, labels, boxes = next(loader_iter)
            except:
                loader_iter = iter(loader)
                image_ids, images, labels, boxes = next(loader_iter)

            # Propagate forward
            logits, bcm = model(images.to(device))
            
            #Prepare Labels
            _, C, H, W = bcm.shape

            if FR_Shifting: #Filling Rate Gudied Learning.
                labels = make_FR_labels(labels, boxes, logits[-1], bcm, size=(H, W))
            else:
                labels = resize_labels(labels, size=(H, W), to_one_hot=False)
            boxes = resize_labels(boxes, size=(C, H, W), to_one_hot=True).to(device)
            
            # Loss
            iter_loss = 0
            iter_loss_seg = 0
            iter_loss_bcm = 0
            for logit in logits:
                # Resize labels for {100%, 75%, 50%, Max} logits
                _, _, H, W = logit.shape
                labels_ = resize_labels(labels, size=(H, W))
                iter_loss_seg += CEL(logit, labels_.to(device))
            iter_loss_bcm = MSEL(bcm, boxes) #Define the BCM loss.
            preds = torch.argmax(logits[-1], dim=1)
            iter_accuracy = float(torch.eq(preds, labels.to(device)).sum().cpu()) / (len(image_ids) * bcm.shape[2] * bcm.shape[3])
            # Propagate backward (just compute gradients)
            iter_loss_seg /= CONFIG.SOLVER.ITER_SIZE
            iter_loss_bcm /= CONFIG.SOLVER.ITER_SIZE
            iter_loss = iter_loss_seg + 0.1*iter_loss_bcm #Overall loss.
            iter_loss.backward()

            loss += float(iter_loss)

        average_loss.add(loss)

        # Update weights with accumulated gradients
        optimizer.step()

        # Update learning rate
        scheduler.step(epoch=iteration)
        cur_time = strftime("%Y-%m-%d %H:%M:%S", localtime())
        log_str = cur_time + ' ' +'iters:{:4}, loss_seg:{:6,.4f}, loss_bcm:{:6,.4f}, accuracy:{:5,.4f}, LR:{:9,.8f}'.format(iteration, iter_loss_seg, iter_loss_bcm, iter_accuracy, optimizer.param_groups[0]["lr"])
        log_file.write(log_str + '\n')
        log_file.flush()
        # TensorBoard
        if iteration % CONFIG.SOLVER.ITER_TB == 0:
            writer.add_scalar("loss/train", average_loss.value()[0], iteration)
            for i, o in enumerate(optimizer.param_groups):
                writer.add_scalar("lr/group_{}".format(i), o["lr"], iteration)
            for i in range(torch.cuda.device_count()):
                writer.add_scalar(
                    "gpu/device_{}/memory_cached".format(i),
                    torch.cuda.memory_cached(i) / 1024 ** 3,
                    iteration,
                )

            if False:
                for name, param in model.module.base.named_parameters():
                    name = name.replace(".", "/")
                    # Weight/gradient distribution
                    writer.add_histogram(name, param, iteration, bins="auto")
                    if param.requires_grad:
                        writer.add_histogram(
                            name + "/grad", param.grad, iteration, bins="auto"
                        )

        # Save a model
        if iteration % CONFIG.SOLVER.ITER_SAVE == 0:
            torch.save(
                model.module.state_dict(),
                os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(iteration)),
            )

    torch.save(
        model.module.state_dict(), os.path.join(checkpoint_dir, "checkpoint_final.pth")
    )
    log_file.close()


@main.command()
@click.option(
    "-c",
    "--config-path",
    type=click.File(),
    required=True,
    help="Dataset configuration file in YAML",
)
@click.option(
    "-m",
    "--model-path",
    type=click.Path(exists=True),
    required=True,
    help="PyTorch model to be loaded",
)
@click.option(
    "--cuda/--cpu", default=True, help="Enable CUDA if available [default: --cuda]"
)

@click.option(
    "--gen-training", default=False, help="Generating training prediction [default: False]"
)
def test(config_path, model_path, cuda, gen_training):
    """
    Evaluation on validation set
    """

    # Configuration
    CONFIG = OmegaConf.load(config_path)
    device = get_device(cuda)
    torch.set_grad_enabled(False)

    # Dataset
    if not gen_training:
        dataset = get_dataset(CONFIG.DATASET.NAME)(
            root=CONFIG.DATASET.ROOT,
            split=CONFIG.DATASET.SPLIT.VAL,
            ignore_label=CONFIG.DATASET.IGNORE_LABEL,
            mean_bgr=(CONFIG.IMAGE.MEAN.B, CONFIG.IMAGE.MEAN.G, CONFIG.IMAGE.MEAN.R),
            augment=False,
        )
    else:
        dataset = get_dataset(CONFIG.DATASET.NAME)(
            root=CONFIG.DATASET.ROOT,
            split='trainaug',
            ignore_label=CONFIG.DATASET.IGNORE_LABEL,
            mean_bgr=(CONFIG.IMAGE.MEAN.B, CONFIG.IMAGE.MEAN.G, CONFIG.IMAGE.MEAN.R),
            augment=False,
        )
    print(dataset)

    # DataLoader
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=CONFIG.SOLVER.BATCH_SIZE.TEST,
        num_workers=CONFIG.DATALOADER.NUM_WORKERS,
        shuffle=False,
    )

    # Model
    model = eval(CONFIG.MODEL.NAME)(n_classes=CONFIG.DATASET.N_CLASSES)
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model = nn.DataParallel(model)
    model.eval()
    model.to(device)

    # Path to save logits
    logit_dir = os.path.join(
        CONFIG.EXP.OUTPUT_DIR,
        "features",
        "logit",
        CONFIG.EXP.NAME,
    )
    makedirs(logit_dir)
    print("Logit dst:", logit_dir)

    # Path to save scores
    save_dir = os.path.join(
        CONFIG.EXP.OUTPUT_DIR,
        "scores",
        CONFIG.EXP.ID,
    )
    makedirs(save_dir)
    save_path = os.path.join(save_dir, 'scores_'+CONFIG.EXP.NAME+'_%s'%datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'.json')
    print("Score dst:", save_path)


    preds, gts = [], []
    for image_ids, images, gt_labels, boxes in tqdm(
        loader, total=len(loader), dynamic_ncols=True
    ):
        # Image
        images = images.to(device)

        # Forward propagation
        logits, bcm = model(images)
        # Save on disk for CRF post-processing
        for image_id, logit in zip(image_ids, logits):
            filename = os.path.join(logit_dir, image_id + ".npy")
            np.save(filename, logit.cpu().numpy())

        # Pixel-wise labeling
        _, H, W = gt_labels.shape
        logits = F.interpolate(
            logits, size=(H, W), mode="bilinear", align_corners=False
        )
        probs = F.softmax(logits, dim=1)
        labels = torch.argmax(probs, dim=1)

        preds += list(labels.cpu().numpy())
        gts += list(gt_labels.numpy())

    # Pixel Accuracy, Mean Accuracy, Class IoU, Mean IoU, Freq Weighted IoU
    score = scores(gts, preds, n_class=CONFIG.DATASET.N_CLASSES)

    with open(save_path, "w") as f:
        json.dump(score, f, indent=4, sort_keys=True)


@main.command()
@click.option(
    "-c",
    "--config-path",
    type=click.File(),
    required=True,
    help="Dataset configuration file in YAML",
)

def gen(config_path):
    """
    Generating pseudo labels.
    """
    # Class palette for test.
    gen_mask  = True
    gen_prob_mask  = True
    palette = []
    for i in range(256):
        palette.extend((i,i,i))
    palette[:3*21] = np.array([[0, 0, 0],
                            [128, 0, 0],
                            [0, 128, 0],
                            [128, 128, 0],
                            [0, 0, 128],
                            [128, 0, 128],
                            [0, 128, 128],
                            [128, 128, 128],
                            [64, 0, 0],
                            [192, 0, 0],
                            [64, 128, 0],
                            [192, 128, 0],
                            [64, 0, 128],
                            [192, 0, 128],
                            [64, 128, 128],
                            [192, 128, 128],
                            [0, 64, 0],
                            [128, 64, 0],
                            [0, 192, 0],
                            [128, 192, 0],
                            [0, 64, 128]], dtype='uint8').flatten()
    palette[-3:] = np.array([[224,224,192]], dtype='uint8').flatten()
    # Configuration
    CONFIG = OmegaConf.load(config_path)
    torch.set_grad_enabled(False)

    # Dataset
    dataset = get_dataset(CONFIG.DATASET.NAME)(
        root=CONFIG.DATASET.ROOT,
        split=CONFIG.DATASET.SPLIT.TRAIN,
        ignore_label=CONFIG.DATASET.IGNORE_LABEL,
        mean_bgr=(CONFIG.IMAGE.MEAN.B, CONFIG.IMAGE.MEAN.G, CONFIG.IMAGE.MEAN.R),
        augment=False,
    )
    print(dataset)

    # CRF post-processor
    postprocessor = DenseCRF(
        iter_max=CONFIG.CRF.ITER_MAX,
        pos_xy_std=CONFIG.CRF.POS_XY_STD,
        pos_w=CONFIG.CRF.POS_W,
        bi_xy_std=CONFIG.CRF.BI_XY_STD,
        bi_rgb_std=CONFIG.CRF.BI_RGB_STD,
        bi_w=CONFIG.CRF.BI_W,
    )

    # Path to logit files
    logit_dir = os.path.join(
        CONFIG.EXP.OUTPUT_DIR,
        "features",
        "logit",
        CONFIG.EXP.NAME,
    )
    pre_fix = CONFIG.EXP.NAME+'_95'
    if gen_mask:
        mask_dir = os.path.join(
            CONFIG.EXP.OUTPUT_DIR,
            "gen_labels",
            pre_fix,
            "mask",
        )
        im_dir = os.path.join(
            CONFIG.EXP.OUTPUT_DIR,
            "gen_labels",
            pre_fix,
            "im",
        )
        seed_dir = os.path.join(
            CONFIG.EXP.OUTPUT_DIR,
            "gen_labels",
            pre_fix,
            "seed",
        )
        box_dir = os.path.join(
            CONFIG.EXP.OUTPUT_DIR,
            "gen_labels",
            pre_fix,
            "box",
        )
        if not os.path.exists(mask_dir):
            makedirs(mask_dir)
        if not os.path.exists(im_dir):
            makedirs(im_dir)
        if not os.path.exists(seed_dir):
            makedirs(seed_dir)
        if not os.path.exists(box_dir):
            makedirs(box_dir)
    if gen_prob_mask:
        mask_prob_dir = os.path.join(
            CONFIG.EXP.OUTPUT_DIR,
            "gen_labels",
            pre_fix,
            "mask_prob",
        )
        if not os.path.exists(mask_prob_dir):
            makedirs(mask_prob_dir)
    
    print("Logit src:", logit_dir)
    if not os.path.isdir(logit_dir):
        print("Logit not found, run first: python main.py test [OPTIONS]")
        quit()
    # Process per sample.
    preds = []
    gts = []
    for i in range(len(dataset)):
        image_id, image, gt_label, box = dataset.__getitem__(i)

        filename = os.path.join(logit_dir, image_id + ".npy")
        logit = np.load(filename)

        _, H, W = image.shape
        logit = torch.FloatTensor(logit)[None, ...]
        logit = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)
        probs = F.softmax(logit, dim=1)[0].numpy()

        label = np.argmax(probs, axis=0)
        if gen_mask: #The original pseudo masks, only for evaluating models.
            img_label = Image.fromarray(label.astype(np.uint8))
            maskname = os.path.join(mask_dir, image_id + ".png")
            img_label.putpalette(palette)
            img_label.save(maskname)

        if gen_prob_mask: #The final pseudo masks, for further training.
            probs = np.max(probs, axis=0)
            gt_label[probs>0.95]= label[probs>0.95]#Set the thresholds, only the high-confident pixels will be updated.
            img_label = Image.fromarray(gt_label.astype(np.uint8))
            maskname = os.path.join(mask_prob_dir, image_id + ".png")
            img_label.putpalette(palette)
            img_label.save(maskname)

            # For better comparison with original data.
            os.system('cp %s %s' % (os.path.join(CONFIG.DATASET.ROOT,'VOC2012','JPEGImages', image_id + '.jpg'), os.path.join(im_dir, image_id + ".jpg")))
            os.system('cp %s %s' % (os.path.join(CONFIG.DATASET.ROOT,'VOC2012','SegmentationClassAug_MG', image_id + '.png'), os.path.join(seed_dir, image_id + ".png")))
            os.system('cp %s %s' % (os.path.join(CONFIG.DATASET.ROOT,'VOC2012','SegmentationClassAug_Box', image_id + '.png'), os.path.join(box_dir, image_id + ".png")))

        print('---->%d of %d'%(i, len(dataset)))


if __name__ == "__main__":
    main()
