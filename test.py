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
from fqcdtrans_arch import *
from fqcdtrans_utils import *
from generate_data import generate_data

import datetime

PATH = os.getcwd()
HOME = '/home/sehyunpark/PSAQ_FQCDTrans/log/'
project_name = 'PSAQ_FQCDTrans'
# Supports: ['int8', 'int4', 'int2', 'uint8', 'uint4', 'uint2]
# args.weights = 'int8'
# args.activation = 'uint8'
# args.attn = 'uint8'


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
        "--model_name", default='ar2cl', help="CDTrans domain adaptation", type=str)
    parser.add_argument(
        "--calibration_mode", default='PSAQ', choices=['PSAQ', 'trainset', 'valset', 'gaussian'], \
            help="mode of calibration set for PTQ", type=str)
    parser.add_argument(
        "--weights", default='int8', help="quantization bit for weights", type=str)
    parser.add_argument(
        "--activation", default='uint8', help="quantization bit for activation", type=str)
    parser.add_argument(
        "--attn", default='uint8', help="quantization bit for attention", type=str)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.freeze()
    
    #Time 
    now = datetime.datetime.utcnow()
    time_gap = datetime.timedelta(hours=9)
    now += time_gap
    now = now.strftime("%Y%m%d_%H%M%S") 
    
    # model_name = cfg.MODEL.MODE
    model_name = args.model_name
    quantization = args.weights[-1] + '_' + args.activation[-1] + '_' + args.attn[-1]
    
    if not os.path.exists(f"{HOME}/{quantization}/{args.calibration_mode}/{model_name}"):
        os.makedirs(f"{HOME}/{quantization}/{args.calibration_mode}/{model_name}")
    # For storing checkpoint after training
    # if not os.path.exists(f"./{project_name}/checkpoint/{model_name}/"):
    #     os.makedirs(f"./{project_name}/checkpoint/{model_name}/")
        
    log_path = f"{HOME}/{quantization}/{args.calibration_mode}/{model_name}/{now}.log"
    log, log_close = create_log_func(log_path)          # Abbreviate log_func to log
    with open(log_path, 'w+') as f:
        pprint.pprint(f)

    log('******************************          Specifications          ******************************')
    log(f'Project name: {project_name}, at time {now}')
    log(f'Model: {model_name}')
    log(f'Quantization Bits for {project_name}')
    log(f'weights: {args.weights}, activation: {args.activation}, attn: {args.attn}')
    log(f'Calibration Mode: {args.calibration_mode}')

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
        num_classes_dataset=65,
        num_classes=1000,
        embed_dim=384,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(QIntLayerNorm, eps=1e-6),
        quant=True,
        calibrate=False,
        input_quant=True,
        BIT_W=args.weights,
        BIT_A=args.activation,
        BIT_S=args.attn)
    
    model_fq = FQCDTrans(
        dataset='office_home',
        num_classes_dataset=65,
        num_classes=1000,
        embed_dim=384,
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
    model_fq.load_state_dict(new_state_dict, strict=False)
    
    # Setting Calibration Loader for PSAQ
    if args.calibration_mode == "PSAQ":
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
        log(f'Calibration Mode: {args.calibration_mode}')
        model.model_open_calibrate()
        with torch.no_grad():
            if args.calibration_mode == "PSAQ":
                for i, image in enumerate(calibrate_loader):
                    model.model_open_last_calibrate()
                    image = image.to(device)
                    _, output, _ = model(None, image, target_branch_only=True)
            elif args.calibration_mode == "trainset":
                for i, (image, _, _, _, _) in enumerate(train_loader): # img: [64, 3, 224, 224] -> train loader batch: 32 -> Just use one batch
                    if i == 0:
                        model.model_open_last_calibrate()
                        image = image.to(device)
                        _, output, _ = model(None, image, target_branch_only=True)
                        break
            elif args.calibration_mode == "valset":
                for i, (image, _, _, _, _, _) in enumerate(val_loader): 
                # if i == batch_random:
                    if i == 0:
                        model.model_open_last_calibrate()
                        image = image.to(device)
                        _, output, _ = model(None, image, target_branch_only=True)
                        break
            elif args.calibration_mode == "gaussian":
                calibrate_data = torch.randn((32, 3, 224, 224))
                model.model_open_last_calibrate()
                image = calibrate_data.to(device)
                _, output, _ = model(None, image, target_branch_only=True)
        model.model_close_calibrate()
        model.model_quant()
        
    log("Validating.....")
    accuracy = do_inference_uda(cfg, model, val_loader, num_query)
    log("Classify Domain Adapatation Validation Results - In the source trained model")
    log(f'weights: {args.weights}, activation: {args.activation}, attn: {args.attn}')
    log(f'Calibration Mode: {args.calibration_mode}')
    log(f'Accuracy: {accuracy}')
    log("Accuracy (2 d.p): {:.2%}".format(accuracy))

