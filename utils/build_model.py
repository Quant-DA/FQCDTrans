from types import MethodType
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models import vision_transformer
from timm.models.vision_transformer import Attention
from timm.models.swin_transformer import WindowAttention

'''
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
        q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]

        attn2 = torch.matmul(q2, k2.transpose(-2, -1))
        attn2 = attn2 * self.scale
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.drop_attn(attn2)

        x2 = torch.matmul(attn2, v2)
        x2 = x2.transpose(1, 2).reshape(B, N, C)
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
'''

def attention_forward(self, x):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

    # attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = self.matmul1(q, k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    del q, k

    # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.matmul2(attn, v).transpose(1, 2).reshape(B, N, C)
    del attn, v
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def window_attention_forward(self, x, mask = None):
    B_, N, C = x.shape
    qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

    q = q * self.scale
    # attn = (q @ k.transpose(-2, -1))
    attn = self.matmul1(q, k.transpose(-2,-1))

    relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    attn = attn + relative_position_bias.unsqueeze(0)

    if mask is not None:
        nW = mask.shape[0]
        attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
    else:
        attn = self.softmax(attn)

    attn = self.attn_drop(attn)

    # x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
    x = self.matmul2(attn, v).transpose(1, 2).reshape(B_, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


class MatMul(nn.Module):
    def forward(self, A, B):
        return A @ B


def build_model(name, Pretrained=True):
    """
    Get a vision transformer model.
    This will replace matrix multiplication operations with matmul modules in the model.

    Currently support almost all models in timm.models.transformers, including:
    - vit_tiny/small/base/large_patch16/patch32_224/384,
    - deit_tiny/small/base(_distilled)_patch16_224,
    - deit_base(_distilled)_patch16_384,
    - swin_tiny/small/base/large_patch4_window7_224,
    - swin_base/large_patch4_window12_384

    These models are finetuned on imagenet-1k and should use ViTImageNetLoaderGenerator
    for calibration and testing.
    """
    net = timm.create_model(name, pretrained=Pretrained)

    for name, module in net.named_modules():
        if isinstance(module, Attention):
            setattr(module, "matmul1", MatMul())
            setattr(module, "matmul2", MatMul())
            module.forward = MethodType(attention_forward, module)
        if isinstance(module, WindowAttention):
            setattr(module, "matmul1", MatMul())
            setattr(module, "matmul2", MatMul())
            module.forward = MethodType(window_attention_forward, module)

    net = net.cuda()
    net.eval()
    return net
