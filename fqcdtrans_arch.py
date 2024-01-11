import math
import torch
import torch.nn as nn
from functools import partial

from fqcdtrans_quant import *
from fqcdtrans_utils import *

"""
BitType
"""
BIT_TYPE_LIST = [
    BitType(4, False, 'uint4'),
    BitType(8, True, 'int8'),
    BitType(4, True, 'int4'),
    BitType(8, False, 'uint8'),
    BitType(2, False, 'uint2'),
]
BIT_TYPE_DICT = {bit_type.name: bit_type for bit_type in BIT_TYPE_LIST}

"""
configs
"""
OBSERVER_W = 'minmax'
OBSERVER_A = 'minmax'

QUANTIZER_W = 'uniform'
QUANTIZER_A = 'uniform'
QUANTIZER_A_LN = 'uniform'

CALIBRATION_MODE_W = 'channel_wise'
CALIBRATION_MODE_A = 'layer_wise'
CALIBRATION_MODE_S = 'layer_wise'

INT_SOFTMAX = True
OBSERVER_S = 'minmax'
QUANTIZER_S = 'log2'

INT_NORM = True
OBSERVER_A_LN = 'ptf'
CALIBRATION_MODE_A_LN = 'channel_wise'

"""
Modules with quantization
"""

class QPatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 norm_layer=None,
                 quant=False,
                 calibrate=False,
                 cfg=None,
                 BIT_W='int8',
                 BIT_A='uint4'):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size

        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        

        self.proj = QConv2d(in_chans,
                            embed_dim,
                            kernel_size=patch_size,
                            stride=patch_size,
                            quant=quant,
                            calibrate=calibrate,
                            bit_type=BIT_W,
                            calibration_mode=CALIBRATION_MODE_W,
                            observer_str=OBSERVER_W,
                            quantizer_str=QUANTIZER_W)
        if norm_layer:
            self.qact_before_norm = QAct(
                quant=quant,
                calibrate=calibrate,
                bit_type=BIT_A,
                calibration_mode=CALIBRATION_MODE_A,
                observer_str=OBSERVER_A,
                quantizer_str=QUANTIZER_A)
            self.norm = norm_layer(embed_dim)
            self.qact = QAct(quant=quant,
                             calibrate=calibrate,
                             bit_type=BIT_A,
                             calibration_mode=CALIBRATION_MODE_A,
                             observer_str=OBSERVER_A,
                             quantizer_str=QUANTIZER_A)
        else:
            self.qact_before_norm = nn.Identity()
            self.norm = nn.Identity()
            self.qact = QAct(quant=quant,
                             calibrate=calibrate,
                             bit_type=BIT_A,
                             calibration_mode=CALIBRATION_MODE_A,
                             observer_str=OBSERVER_A,
                             quantizer_str=QUANTIZER_A)

    def forward(self, x):
        # B, C, H, W = x.shape
        B, C, H, W = x.shape # for iteration 
        # FIXME look at relaxing size constraints
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # print(f'x shape: {x.shape}')
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.qact_before_norm(x)
        if isinstance(self.norm, nn.Identity):
            x = self.norm(x)
        else:
            x = self.norm(x, self.qact_before_norm.quantizer,
                          self.qact.quantizer)
        x = self.qact(x)
        return x


class QMlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.0,
                 quant=False,
                 calibrate=False,
                 cfg=None,
                 BIT_W='int8',
                 BIT_A='uint4'):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc1 = QLinear(in_features,
                           hidden_features,
                           quant=quant,
                           calibrate=calibrate,
                           bit_type=BIT_W,
                           calibration_mode=CALIBRATION_MODE_W,
                           observer_str=OBSERVER_W,
                           quantizer_str=QUANTIZER_W)
        self.act = act_layer()
        self.qact1 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=BIT_A,
                          calibration_mode=CALIBRATION_MODE_A,
                          observer_str=OBSERVER_A,
                          quantizer_str=QUANTIZER_A)
        # self.fc2 = nn.Linear(hidden_features, out_features)
        self.fc2 = QLinear(hidden_features,
                           out_features,
                           quant=quant,
                           calibrate=calibrate,
                           bit_type=BIT_W,
                           calibration_mode=CALIBRATION_MODE_W,
                           observer_str=OBSERVER_W,
                           quantizer_str=QUANTIZER_W)
        self.qact2 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=BIT_A,
                          calibration_mode=CALIBRATION_MODE_A,
                          observer_str=OBSERVER_A,
                          quantizer_str=QUANTIZER_A)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.qact1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.qact2(x)
        x = self.drop(x)
        return x

class QCDTransAttention(nn.Module):
    def __init__(
        self, 
        dim, 
        num_heads, 
        qkv_bias, 
        qk_scale, 
        drop_attn_rate, 
        drop_proj_rate,
        quant=False,
        calibrate=False,
        BIT_W='int8',
        BIT_A='uint4',
        BIT_S='uint8'):
        super(QCDTransAttention, self).__init__()
        head_dim = dim // num_heads # 64
        self.num_heads = num_heads 
        self.scale = qk_scale or head_dim**(-0.5) # 0.125 from head_dim**(-0.5)

        self.qkv = QLinear(dim,
                           dim * 3,
                           bias=qkv_bias,
                           quant=quant,
                           calibrate=calibrate,
                           bit_type=BIT_W,
                           calibration_mode=CALIBRATION_MODE_W,
                           observer_str=OBSERVER_W,
                           quantizer_str=QUANTIZER_W)
        self.qact1 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=BIT_A,
                          calibration_mode=CALIBRATION_MODE_A,
                          observer_str=OBSERVER_A,
                          quantizer_str=QUANTIZER_A)
        self.matmul1 = QMatMul()
        self.qact2 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=BIT_A,
                          calibration_mode=CALIBRATION_MODE_A,
                          observer_str=OBSERVER_A,
                          quantizer_str=QUANTIZER_A)
        self.softmax = QIntSoftmax(
            log_i_softmax=INT_SOFTMAX,
            quant=quant,
            calibrate=calibrate,
            bit_type=BIT_S,
            calibration_mode=CALIBRATION_MODE_S,
            observer_str=OBSERVER_S,
            quantizer_str=QUANTIZER_S) # log_int_softmax
        
        self.qact_attn1 = QAct(quant=quant,
                               calibrate=calibrate,
                               bit_type=BIT_A,
                               calibration_mode=CALIBRATION_MODE_A,
                               observer_str=OBSERVER_A,
                               quantizer_str=QUANTIZER_A) #qact2
        
        self.qact3 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=BIT_A,
                          calibration_mode=CALIBRATION_MODE_A,
                          observer_str=OBSERVER_A,
                          quantizer_str=QUANTIZER_A) 
        
        self.matmul2 = QMatMul()
        
        self.proj = QLinear(dim,
                            dim,
                            quant=quant,
                            calibrate=calibrate,
                            bit_type=BIT_W,
                            calibration_mode=CALIBRATION_MODE_W,
                            observer_str=OBSERVER_W,
                            quantizer_str=QUANTIZER_W)

        
        self.drop_attn = nn.Dropout(drop_attn_rate)
        self.drop_proj = nn.Dropout(drop_proj_rate)

    def forward(self, x1, x2, target_branch_only=True):
        """
        1: source branch (H_S)
        2: target branch (H_T)
        3: source-target branch (H_{S+T})
        """
        B, N, C = x2.shape

        if target_branch_only:
            assert x1 is None

            qkv2 = self.qkv(x2)
            x = self.qact1(qkv2)
            qkv2 = qkv2.reshape(B, N, 3, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)
            q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2] # cannot use tensor as tuple

            # attn2 = torch.matmul(q2, k2.transpose(-2, -1)) # For CDTrans
            # attn2 = self.matmul1(q2, k2.transpose(-2, -1)) # For PSAQ-ViT
            attn2 = (q2 @ k2.transpose(-2, -1)) * self.scale
            attn2 = self.qact_attn1(attn2) 
            # attn = self.log_int_softmax(attn, self.qact_attn1.quantizer.scale) # FQ-ViT
            attn2 = self.softmax(attn2, self.qact_attn1.quantizer.scale)
            attn2 = self.drop_attn(attn2)

            del q2, k2

            # x2 = torch.matmul(attn2, v2) # For CDTrans
            # x2 = self.matmul2(attn2, v2) # For PSAQ-ViT
            x2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, C)
            del attn2, v2
            x2 = self.qact2(x2)
            x2 = self.proj(x2)
            x2 = self.qact3(x2)
            x2 = self.drop_proj(x2)

            x3 = None
        else:
            qkv1 = self.qkv(x1)
            x1 = self.qact1(qkv1)
            qkv1 = qkv1.reshape(B, N, 3, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)
            q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2]

            qkv2 = self.qkv(x2)
            x2 = self.qact1(qkv2)
            qkv2 = qkv2.reshape(B, N, 3, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)
            q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]

            attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale
            attn1 = self.qact_attn1(attn1)
            attn1 = self.softmax(attn1, self.qact_attn1.quantizer.scale)
            attn1 = self.drop_attn(attn1)
            
            attn2 = (q2 @ k2.transpose(-2, -1)) * self.scale
            attn2 = self.qact_attn1(attn2)
            attn2 = self.softmax(attn2, self.qact_attn1.quantizer.scale)
            attn2 = self.drop_attn(attn2)
            
            attn3 = (q1 @ k2.transpose(-2, -1)) * self.scale
            attn3 = self.qact_attn1(attn3)
            attn3 = self.softmax(attn3, self.qact_attn1.quantizer.scale)
            attn3 = self.drop_attn(attn3)

            x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C)
            # del attn1, v1
            x1 = self.qact2(x1)
            x1 = self.proj(x1)
            x1 = self.qact3(x1)
            x1 = self.drop_proj(x1)
            
            x2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, C)
            del attn2, v2
            x2 = self.qact2(x2)
            x2 = self.proj(x2)
            x2 = self.qact3(x2)
            x2 = self.drop_proj(x2)
            
            x3 = (attn3 @ v2).transpose(1, 2).reshape(B, N, C)
            del attn2, v2
            x3 = self.qact2(x3)
            x3 = self.proj(x3)
            x3 = self.qact3(x3)
            x3 = self.drop_proj(x3)
        return x1, x2, x3
        

class QCDTransBlock(nn.Module):
    def __init__(
        self, 
        dim, 
        num_heads, 
        mlp_ratio=4.0, 
        qkv_bias=False, 
        qk_scale=None, 
        drop_rate=0.0, #drop 
        drop_attn_rate=0.0, #attn_drop
        drop_path_rate=0.0, #drop_path
        act_layer=nn.GELU, 
        norm_layer=nn.LayerNorm,
        quant=False,
        calibrate=False,
        BIT_W='int8',
        BIT_A='uint8',
        BIT_S='uint8'
        ):
        super(QCDTransBlock, self).__init__()
        
        self.norm1 = norm_layer(dim)
        self.qact1 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=BIT_A,
                          calibration_mode=CALIBRATION_MODE_A,
                          observer_str=OBSERVER_A,
                          quantizer_str=QUANTIZER_A)
        self.attn = QCDTransAttention(dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              drop_attn_rate=drop_attn_rate,
                              drop_proj_rate=drop_rate,
                              BIT_W=BIT_W,
                              BIT_A=BIT_A,
                              BIT_S=BIT_S)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.qact_pos = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=BIT_A,
                          calibration_mode=CALIBRATION_MODE_A_LN,
                          observer_str=OBSERVER_A_LN,
                          quantizer_str=QUANTIZER_A_LN) #qact2 = qact_pos
        self.norm2 = norm_layer(dim)
        self.qact3 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=BIT_A,
                          calibration_mode=CALIBRATION_MODE_A,
                          observer_str=OBSERVER_A,
                          quantizer_str=QUANTIZER_A)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = QMlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop_rate,
                       quant=quant,
                       calibrate=calibrate,
                       BIT_W=BIT_W,
                       BIT_A=BIT_A)
        self.qact4 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=BIT_A,
                          calibration_mode=CALIBRATION_MODE_A_LN,
                          observer_str=OBSERVER_A_LN,
                          quantizer_str=QUANTIZER_A_LN)


    def forward(self, x1, x2, x3, target_branch_only=True, last_quantizer=None):
        """
        1: source branch (H_S), input
        2: target branch (H_T), input
        3: source-target branch (H_{S+T}), input

        a: source branch (H_S), output
        b: target branch (H_T), output
        ab: source-target branch (H_{S+T}), output
        """
        if target_branch_only:
            assert (x1 is None) and (x3 is None) 

            x2_ = self.norm1(x2, last_quantizer, self.qact1.quantizer)
            x2_ = self.qact1(x2_)
            _, xa_attn2, _ = self.attn(None, x2_, target_branch_only)
            
            xa_attn2 = self.drop_path(xa_attn2)
            xb = self.qact_pos(x2 + xa_attn2)
            xb_= self.norm2(xb, self.qact_pos.quantizer, self.qact3.quantizer)
            xb_ = self.qact3(xb_)
            xb_ = self.mlp(xb_)
            xb_ = self.drop_path(xb_)
            xb = self.qact4(xb_ + xb)         

            xa, xab = None, None
        else:
            x1_ = self.norm1(x1, self.qact1.quantizer)
            x1_ = self.qact1(x1_)

            x2_ = self.norm1(x2, self.qact1.quantizer)
            x2_ = self.qact1(x2_)

            xa_attn1, xa_attn2, xa_attn3 = self.attn(x1_,x2_, target_branch_only)
            
            xa_attn1 = self.drop_path(xa_attn1)
            xa = self.qact_pos(x1 + xa_attn1)
            xa_ = self.norm2(xa, self.qact_pos.quantizer, self.qact3.quantizer)
            xa_ = self.qact3(xa_)
            xa_ = self.mlp(xa_)
            xa_ = self.drop_path(xa_)
            xa = self.qact4(xa_ + xa) 
            

            xa_attn2 = self.drop_path(xa_attn2)
            xb = self.qact_pos(x2 + xa_attn2)
            xb_= self.norm2(xb, self.qact_pos.quantizer, self.qact3.quantizer)
            xb_ = self.qact3(xb_)
            xb_ = self.mlp(xb_)
            xb_ = self.drop_path(xb_)
            xb = self.qact4(xb_ + xb)  
            
            xa_attn3 = self.drop_path(xa_attn3)
            xab = self.qact_pos(x3 + xa_attn3)
            xab_= self.norm2(xab, self.qact_pos.quantizer, self.qact3.quantizer)
            xab_ = self.qact3(xab_)
            xab_ = self.mlp(xab_)
            xab_ = self.drop_path(xab_)
            xab = self.qact4(xab_ + xab)  
        return xa, xb, xab

"""
Models
"""
class FQCDTrans(nn.Module):
    def __init__(
        self,
        dataset='office_home',
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000, # Head For ImageNet 
        num_classes_dataset=65,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        drop_attn_rate=0.0,
        last_drop_path_rate=0.0,
        act_layer=None,
        norm_layer=None,
        # for fqvit(PTQ)
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        quant=False,
        calibrate=False,
        input_quant=False,
        BIT_W='int8',
        BIT_A='uint8',
        BIT_S='uint8'):
        super().__init__()
        act_layer = act_layer or nn.GELU
        norm_layer = norm_layer or nn.LayerNorm
        dprs = [x.item() for x in torch.linspace(0, last_drop_path_rate, depth)]  # Stochastic depth decay rule
        self.dataset = dataset
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim    

        # for fqvit(PTQ)
        self.input_quant = input_quant
        if input_quant:
            self.qact_input = QAct(quant=quant,
                                   calibrate=calibrate,
                                   # last_calibrate=False,
                                   bit_type=BIT_A,
                                   calibration_mode=CALIBRATION_MODE_A,
                                   observer_str=OBSERVER_A,
                                   quantizer_str=QUANTIZER_A)
        self.patch_embed = QPatchEmbed(img_size=img_size,
                                        patch_size=patch_size,
                                        in_chans=in_chans,
                                        embed_dim=embed_dim,
                                        quant=quant,
                                        calibrate=calibrate,
                                        BIT_W=BIT_W,
                                        BIT_A=BIT_A)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        self.qact_embed = QAct(quant=quant,
                               calibrate=calibrate,
                               bit_type=BIT_A,
                               calibration_mode=CALIBRATION_MODE_A,
                               observer_str=OBSERVER_A,
                               quantizer_str=QUANTIZER_A) # embedding activation
        self.qact_pos = QAct(quant=quant,
                             calibrate=calibrate,
                             bit_type=BIT_A,
                             calibration_mode=CALIBRATION_MODE_A,
                             observer_str=OBSERVER_A,
                             quantizer_str=QUANTIZER_A) # positional activation
        self.qact1 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=BIT_A,
                          calibration_mode=CALIBRATION_MODE_A_LN,
                          observer_str=OBSERVER_A_LN,
                          quantizer_str=QUANTIZER_A_LN)
        self.blocks = nn.ModuleList([
            QCDTransBlock(dim=embed_dim,
                  num_heads=num_heads,
                  mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias,
                  qk_scale=qk_scale,
                  drop_rate=drop_rate,
                  drop_attn_rate=attn_drop_rate,
                  drop_path_rate=dprs[i],
                  act_layer = act_layer,
                  norm_layer=norm_layer,
                  quant=quant,
                  calibrate=calibrate,
                  BIT_W=BIT_W,
                  BIT_A=BIT_A,
                  BIT_S=BIT_S) for i in range(depth)],)
        self.norm = norm_layer(embed_dim)
        self.qact2 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=BIT_A,
                          calibration_mode=CALIBRATION_MODE_A,
                          observer_str=OBSERVER_A,
                          quantizer_str=QUANTIZER_A)
        self.head = (QLinear(self.num_features,
                             num_classes,
                             quant=quant,
                             calibrate=calibrate,
                             bit_type=BIT_W,
                             calibration_mode=CALIBRATION_MODE_W,
                             observer_str=OBSERVER_W,
                             quantizer_str=QUANTIZER_W)
                     if num_classes > 0 else nn.Identity())
        self.act_out = QAct(quant=quant,
                            calibrate=calibrate,
                            bit_type=BIT_A,
                            calibration_mode=CALIBRATION_MODE_A,
                            observer_str=OBSERVER_A,
                            quantizer_str=QUANTIZER_A)        # Classifier for officehome
        self.classifier = (QLinear(self.embed_dim,
                             num_classes_dataset,
                             quant=quant,
                             calibrate=calibrate,
                             bit_type=BIT_W,
                             calibration_mode=CALIBRATION_MODE_W,
                             observer_str=OBSERVER_W,
                             quantizer_str=QUANTIZER_W)
                     if num_classes_dataset > 0 else nn.Identity()) # For classifier of OfficeHome

        no_grad_trunc_normal(self.pos_embed, std=0.02)
        no_grad_trunc_normal(self.cls_token, std=0.02)
        self.apply(self._init_params)

    def _init_params(self, m):
        if isinstance(m, nn.Linear):
            no_grad_trunc_normal(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    # def get_classifier(self):
    #     return self.head

    # def reset_classifier(self, num_classes, global_pool=''):
    #     self.num_classes = num_classes
    #     self.head = (nn.Linear(self.embed_dim, num_classes)
                    #  if num_classes > 0 else nn.Identity())

    def model_quant(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.quant = True
            # if self.cfg.INT_NORM:
            #     if type(m) in [QIntLayerNorm]:
            #         m.mode = 'int'
            elif type(m) in [QIntLayerNorm]:
                m.mode = 'int'

    def model_dequant(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.quant = False

    def model_open_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.calibrate = True
                
    def model_open_last_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.last_calibrate = True

    def model_close_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.calibrate = False

    def forward_features(self, x1, x2, target_branch_only):
        B = x2.shape[0] # Batch

        if target_branch_only:
            assert x1 is None
            
            if self.input_quant:
                # print("qact.", self.qact_input.quant)
                x2 = self.qact_input(x2)

            # x2 = self.qact1(x2)
            x2 = self.patch_embed(x2)

            cls_tokens = self.cls_token.expand(B, -1, -1) 
            x2 = torch.cat((cls_tokens, x2), dim=1)
            x2 = self.qact_embed(x2)
            x2 = x2 + self.qact_pos(self.pos_embed)
            x2 = self.qact1(x2)
            x2 = self.pos_drop(x2)

            for i, block in enumerate(self.blocks):
                last_quantizer = self.qact1.quantizer if i ==0 else self.blocks[i-1].qact4.quantizer
                _, x2, _ = block(None, x2, None, target_branch_only, last_quantizer)                

            x2 = self.norm(x2, self.blocks[-1].qact4.quantizer, self.qact2.quantizer)
            x2 = x2[:, 0]
            x2 = self.qact2(x2)
            
            return None, x2, None
        
        else:
            assert B == x1.shape[0]

            if self.input_quant:
                # print("qact.", self.qact_input.quant)
                x1 = self.qact_input(x1)
                x2 = self.qact_input(x2)

            # x1 = self.qact1(x1)
            x1 = self.patch_embed(x1)

            # x2 = self.qact1(x2)
            x2 = self.patch_embed(x2)

            cls_tokens = self.cls_token.expand(B, -1, -1)
            x1 = torch.cat((cls_tokens, x1), dim=1)
            x_pos = self.qact_pos(self.pos_embed)
            x1 = self.qact1(x1, x_pos)
            x1 = self.pos_drop(x1)

            x2 = torch.cat((cls_tokens, x2), dim=1)
            x_pos = self.qact_pos(self.pos_embed)
            x2 = self.qact1(x2, x_pos)
            x2 = self.pos_drop(x2)

            x3 = x2
            for block in self.blocks:
                x1, x2, x3 = block(x1, x2, x3, target_branch_only)

            x1 = self.norm(x1)
            x1 = x1[:, 0]
            x1 = self.qact2(x1)

            x2 = self.norm(x2)
            x2 = x2[:, 0]
            x2 = self.qact2(x2)

            x3 = self.norm(x3)
            x3 = x3[:, 0]
            x3 = self.qact2(x3)
            
            return x1, x2, x3

    def forward(self, x1, x2, target_branch_only):
        if target_branch_only:
            assert x1 is None

            _, x2, _ = self.forward_features(None, x2, target_branch_only)
            if self.dataset == 'imagenet': x2 = self.head(x2)
            elif self.dataset == 'office_home': 
                # print(f'dataset: {self.dataset}')
                x2 = self.classifier(x2)
                # x2 = self.act_out(x2)
            else: NotImplementedError

            # x2, _ = self.qact5(x2)
            return None, x2, None
        else:
            x1, x2, x3 = self.forward_features(x1, x2, target_branch_only)
            if self.dataset == 'imagenet': x1 = self.head(x1)
            elif self.dataset == 'office_home' or 'visda': 
                x1 = self.classifier(x1)
                # x1 = self.act_out(x1)
            else: NotImplementedError
            # x1, _ = self.qact5(x1)

            if self.dataset == 'imagenet': x2 = self.head(x2)
            elif self.dataset == 'office_home' or 'visda': 
                x2 = self.classifier(x2)
                # x2 = self.act_out(x2)
            else: NotImplementedError

            # x2, _ = self.qact5(x2)

            if self.dataset == 'imagenet': x3 = self.head(x3)
            elif self.dataset == 'office_home' or 'visda': 
                x3 = self.classifier(x3)
                # x3 = self.act_out(x3)
            else: NotImplementedError

            # x3, _ = self.qact5(x3)
            return x1, x2, x3

    def load_distilled_param(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')
        if 'model' in param_dict:
            param_dict = param_dict['model']

        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        
        for k, v in param_dict.items():
            if 'head' in k or 'dist' in k:
                continue
            if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = self.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
                # To resize pos embedding when using model at different size from pretrained weights
                if 'distilled' in model_path:
                    print('distill need to choose right cls token in the pth')
                    v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                v = resize_pos_embed(v, self.pos_embed, self.patch_embed.num_y, self.patch_embed.num_x)
                # self.state_dict()[k].copy_(revise)
            try:
                self.state_dict()[k].copy_(v)
                # print(f"load {k} successfully")
            except:
                print('===========================ERROR=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape, self.state_dict()[k].shape))



# def deit_small_patch16(
#     input_size,
#     dataset,
#     quantized,
#     cdtrans,
#     pretrained,
#     ckpt_dir,
#     **kwargs):

#     if pretrained and ckpt_dir is not None:
#         if 'source_train_val_vit' in ckpt_dir or 'source_train_vit' in ckpt_dir:
#             print(f"Get pretrained checkpoint from {ckpt_dir}")
#             ckpt = torch.load(ckpt_dir)['model']
#             for param in ckpt.keys():
#                 if param not in model.state_dict().keys():
#                     print(f"Param {param} not in model.state_dict().keys()")
#                     continue
#                 else:
#                     model.state_dict()[param].copy_(ckpt[param])
#         elif 'implementation_check_cdtrans' in ckpt_dir:
#             print(f"Get pretrained checkpoint from {ckpt_dir}")
#             try:
#                 model.load_state_dict(torch.load(ckpt_dir)['model'])
#             except Exception as e:
#                 print(e)

#         elif 'implementation_check_ivit' in ckpt_dir:
#             print(f"Get pretrained checkpoint from {ckpt_dir}")
#             ckpt = torch.load(ckpt_dir, map_location = 'cpu')['model'] # , map_location = 'cuda:0'
#             for param in model.state_dict().keys():
#                 if model.state_dict()[param].shape != ckpt[param].shape:
#                     if 'act_scaling_factor' in param and model.state_dict()[param].shape == torch.Size([1]): 
#                         model.state_dict()[param] = model.state_dict()[param].squeeze() # torch.Size([1]) -> torch.Size([])
#                     # elif 'norm_scaling_factor' in param and model.state_dict()[param].shape == torch.Size([1]):
#                     #     model.state_dict()[param] = torch.zeros(384) # torch.Size([1]) -> torch.Size([384])
#             try:
#                 model.load_state_dict(torch.load(ckpt_dir, map_location = 'cpu')['model'])
#             except Exception as e:
#                 print(e)

#         elif 'implementation_check_fqvit' in ckpt_dir:
#             print(f"Get pretrained checkpoint from {ckpt_dir}")
#             try:
#                 model.load_state_dict(torch.load(ckpt_dir)['model'])
#             except Exception as e:
#                 print(e)
#         elif 'distilled' in ckpt_dir: # for imagenet checkpoint from deit_distilled model 
#             print("Get pretrained checkpoint from deit_distilled model")
#             model.load_distilled_param(ckpt_dir)
#         elif 'imagenet' in ckpt_dir:
#             ckpt = torch.hub.load_state_dict_from_url(
#                 url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
#                 map_location="cpu", 
#                 check_hash=True)
#             if ('num_classes' in kwargs.keys()) and (kwargs['num_classes'] != 1000):
#                 for key in list(ckpt['model'].keys()):
#                     if 'head' in key:
#                         del ckpt['model'][key]
#             model.load_state_dict(ckpt['model'], strict=False)
#         else:
#             print('wrong pretrained model')
#     else:
#         print('no pretrained model')
#     return model