import os
from config import cfg
import argparse
import torch
from torchvision import transforms
from PIL import Image
from datasets import make_dataloader
from model import make_model
from processor import do_inference, do_inference_uda
import pprint
from utils.logger import setup_logger
from types import MethodType

from functools import partial
from arch_fq_cd import *
# from utils_fq import MatMul, attention_forward, AttentionMap
from generate_data import generate_data

import datetime
from datetime import timedelta
import os

PATH = os.getcwd()
HOME = '/home/sehyunpark/Quant_Preliminary/'
project_name = 'PSAQ_FQCDTrans_8/8/4'

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

def attention_forward(self, x1, x2, target_branch_only=True): # For CDTrans
    """
    1: source branch (H_S)
    2: target branch (H_T)
    3: source-target branch (H_{S+T})
    """
    B, N, C = x2.shape

    if target_branch_only:
        assert x1 is None

        qkv2 = self.qkv(x2)
        qkv2 = qkv2.reshape(B, N, 3, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)
        q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2] # cannot use tensor as tuple # q2.shape = torch.Size([32, 6, 197, 64])

        # attn2 = torch.matmul(q2, k2.transpose(-2, -1)) # For CDTrans
        attn2 = self.matmul1(q2, k2.transpose(-2, -1)) # For PSAQ-ViT
        attn2 = attn2 * self.scale
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.drop_attn(attn2)
        del q2, k2

        # x2 = torch.matmul(attn2, v2) # For CDTrans
        x2 = self.matmul2(attn2, v2) # For PSAQ-ViT
        x2 = x2.transpose(1, 2).reshape(B, N, C)
        del attn2, v2
        x2 = self.proj(x2)
        x2 = self.drop_proj(x2)

        x3 = None
    else:
        qkv1 = self.qkv(x1)
        qkv1 = qkv1.reshape(B, N, 3, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)
        q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2]

        qkv2 = self.qkv(x2)
        qkv2 = qkv2.reshape(B, N, 3, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)
        q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]

        attn1 = torch.matmul(q1, k1.transpose(-2, -1))
        attn1 = attn1 * self.scale
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.drop_attn(attn1)

        attn2 = torch.matmul(q2, k2.transpose(-2, -1))
        attn2 = attn2 * self.scale
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.drop_attn(attn2)

        attn3 = torch.matmul(q1, k2.transpose(-2, -1))
        attn3 = attn3 * self.scale
        attn3 = attn3.softmax(dim=-1)
        attn3 = self.drop_attn(attn3)

        x1 = torch.matmul(attn1, v1)
        x1 = x1.transpose(1, 2).reshape(B, N, C)
        x1 = self.proj(x1)
        x1 = self.drop_proj(x1)

        x2 = torch.matmul(attn2, v2)
        x2 = x2.transpose(1, 2).reshape(B, N, C)
        x2 = self.proj(x2)
        x2 = self.drop_proj(x2)

        x3 = torch.matmul(attn3, v2)
        x3 = x3.transpose(1, 2).reshape(B, N, C)
        x3 = self.proj(x3)
        x3 = self.drop_proj(x3)
    return x1, x2, x3

class MatMul(nn.Module):
    def forward(self, A, B):
        return A @ B
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FQCDTrans_Test")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str)
    parser.add_argument(
        "--quant", default=True, help="whether to PTQ or not", type=bool)
    parser.add_argument(
        "--calib-batchsize", default=32, help="number of calibration set to generate", type=int)
    parser.add_argument(
        "--n_classes", default=65, help="number of classes in OfficeHome", type=int)
    parser.add_argument(
        "--epochs", default=2, help="number of epochs to generate images", type=int)
    parser.add_argument(
        "--model", default='deit_small', help="baseline model", type=str)
    parser.add_argument(
        "--model_name", default='ar2cl', help="baseline model", type=str)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    cfg.freeze()
    
    #Time 
    now = datetime.datetime.utcnow()
    time_gap = datetime.timedelta(hours=9)
    now += time_gap
    now = now.strftime("%Y%m%d_%H%M%S") 
    
    # model_name = cfg.MODEL.MODE
    model_name = args.model_name
    
    if not os.path.exists(f"./{project_name}/report/{model_name}/"):
        os.makedirs(f"./{project_name}/report/{model_name}/")
    # For storing checkpoint after training
    # if not os.path.exists(f"./{project_name}/checkpoint/{model_name}/"):
    #     os.makedirs(f"./{project_name}/checkpoint/{model_name}/")
        
    log_path = f"./{project_name}/report/{model_name}/{now}.log"
    log, log_close = create_log_func(log_path)          # Abbreviate log_func to log
    with open(log_path, 'w+') as f:
        pprint.pprint(f)

    log('******************************          Specifications          ******************************')
    log(f'Project name: {project_name}, at time {now}')
    log(f'Model: {model_name}')

    if args.config_file != "":
        log("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            log(config_str)
    log("Running with config:\n{}".format(cfg))

    # GPU Setting
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    device = torch.device('cuda:{}'.format(cfg.MODEL.DEVICE_ID)) if torch.cuda.is_available() else torch.device('cpu') 
    torch.cuda.set_device(device) ### Assign GPU
    log(f'Device: {device}')

    # Dataloader
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num, train_loader1, train_loader2, img_num1, img_num2, s_dataset, t_dataset = make_dataloader(cfg)
     
    # Load Model 
    model = FQCDTrans(
        dataset='office_home',
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_classes=1000,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(QIntLayerNorm, eps=1e-6),
        quant=True,
        calibrate=False,
        input_quant=True)
    
    model_fq = FQCDTrans(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(QIntLayerNorm, eps=1e-6),
        quant=False,
        calibrate=False,
        input_quant=False)
    
    # Load Model Checkpoint
    model_ckpt = torch.load(cfg.TEST.WEIGHT, map_location=device)
    # model_ckpt = torch.load('/home/sehyunpark/Quant_Preliminary/checkpoints/officehome_uda_ar2cl_vit_small_384.pth')
    new_state_dict = ckpt_mapping(model_ckpt)       
    model.load_state_dict(new_state_dict, strict=False)
    # model_fq.load_state_dict(new_state_dict, strict=False)
    
    # Generate calibration set
    model_fq.to(device)
    for name, module in model_fq.named_modules():
        if isinstance(module, QCDTransAttention):
            setattr(module, "matmul1", MatMul())
            setattr(module, "matmul2", MatMul())
            module.forward = MethodType(attention_forward, module)

    log("Generating data.....")
    log("Using {} as a model...".format(type(model_fq)))
    calibration_data = generate_data(args, model_fq, name = "CDTrans") # [32, 3, 224, 224]

    # Calibration Set Loader
    log("Set Calibration Set.....")
    # calibration_tensors = calibration_load('/home/sehyunpark/Quant_Preliminary/PSAQ-ViT/CDTrans/calibration_set') # if loading from images
    calibrate_loader = torch.utils.data.DataLoader(
        calibration_data.cpu(),
        batch_size=32,
        shuffle=False,
        num_workers=8,
        pin_memory=True)
    
    # Calibration
    model.to(device)
    model.eval()
    if args.quant:
        log('Calibrating......')
        model.model_open_calibrate()
        with torch.no_grad():
            for i, image in enumerate(calibrate_loader):
                model.model_open_last_calibrate()
                image = image.to(device)
                _, output, _ = model(None, image, target_branch_only=True)
        model.model_close_calibrate()
        model.model_quant()
        
    log("Validating.....")
    accuracy = do_inference_uda(cfg, model, val_loader, num_query)
    print(accuracy)
    log("Classify Domain Adapatation Validation Results - In the source trained model")
    log("Accuracy: {:.1%}".format(accuracy))

