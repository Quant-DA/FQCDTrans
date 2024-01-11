import collections.abc as container_abcs
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from itertools import repeat

def calculate_topk_accs(outputs, targets, topks=(1,)):
    with torch.no_grad():
        max_topk = max(topks)

        _, preds = outputs.topk(max_topk, 1, True, True)
        preds = preds.t()
        corrects = preds.eq(targets.view(1, -1).expand_as(preds))

        accs = []
        for topk in topks:
            acc = corrects[:topk].reshape(-1).float().sum(0, keepdim=True)
            accs.append(acc.mul_(100.0 / targets.size(0)))
        return accs

def create_log_func(path):
    f = open(path, 'a')
    counter = [0]
    def log(txt, color=None):
        f.write(txt + '\n')
        if color == 'red': txt = "\033[91m {}\033[00m" .format(txt)
        elif color == 'green': txt = "\033[92m {}\033[00m" .format(txt)
        elif color == 'violet': txt = "\033[95m {}\033[00m" .format(txt)
        print(txt)
        counter[0] += 1
        if counter[0] % 10 == 0:
            f.flush()
            os.fsync(f.fileno())
    return log, f.close

def no_grad_trunc_normal(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # Refer to https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2.0 * std) or (mean > b + 2.0 * std):
        print("Mean is more than 2 STDs from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.\n")

    with torch.no_grad():
        l = norm_cdf((a - mean)/std)
        u = norm_cdf((b - mean)/std)
        tensor.uniform_(2*l-1, 2*u-1)
        tensor.erfinv_()
        tensor.mul_(std*math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def to_2tuple(x):
    if isinstance(x, container_abcs.Iterable):
        return x
    return tuple(repeat(x, 2))

class AverageMeter(object): # Compute and store average and current value
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

def resize_pos_embed(posemb, posemb_new, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    print('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if True:
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))

    print('Position embedding resize to height:{} width: {}'.format(hight, width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    # posemb_grid = F.interpolate(posemb_grid, size=(width, hight), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb

def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    # Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
    """
    if reduction == 'none':
        return (pred - tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred - tgt).abs().pow(p).mean()
    
def calibration_load(calibration_set='/home/sehyunpark/Quant_Preliminary/PSAQ-ViT/CDTrans/calibration_set'):
    # Transform to convert images to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # List to store the tensors
    calibration_tensors = []

    # Load each image, transform it, and add it to the list
    # 32 images [Batch], img_shape = [3, 224, 224]
    for filename in os.listdir(calibration_set):
        if filename.endswith('.png') or filename.endswith('.jpg'):  # check for image files
            image_path = os.path.join(calibration_set, filename)
            image = Image.open(image_path)
            image_tensor = transform(image)
            calibration_tensors.append(image_tensor)

    calibration_data_tensor = torch.stack(calibration_tensors)
    return calibration_data_tensor

def ckpt_mapping(ckpt):
    key_map = {'base.fc.weight': 'head.weight',
    'base.fc.bias': 'head.bias',}

    # Rename keys in the checkpoint
    new_state_dict = {}
    for key, value in ckpt.items():
        if 'base' in key:
            new_key = key.replace('base.', '')  # Remove 'base.' prefix  
        # new_state_dict[new_key] = value
        if 'base.fc' in key:
            new_key = key_map.get(key, key)  # Replace with new key if in map, else keep same 
        if 'classifier.weight' in key:
            new_key = 'classifier.weight'
        if 'bottleneck' in key:
            new_key = key
        new_state_dict[new_key] = value
    
    return new_state_dict

