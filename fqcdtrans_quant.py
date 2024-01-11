import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
import collections.abc
from itertools import repeat
from fqcdtrans_utils import lp_loss

  
"""
BitType
"""
class BitType:

    def __init__(self, bits, signed, name=None):
        self.bits = bits
        self.signed = signed
        if name is not None:
            self.name = name
        else:
            self.update_name()

    @property
    def upper_bound(self):
        if not self.signed:
            return 2**self.bits - 1
        return 2**(self.bits - 1) - 1

    @property
    def lower_bound(self):
        if not self.signed:
            return 0
        return -(2**(self.bits - 1))

    @property
    def range(self):
        return 2**self.bits

    def update_name(self):
        self.name = ''
        if not self.signed:
            self.name += 'uint'
        else:
            self.name += 'int'
        self.name += '{}'.format(self.bits)

BIT_TYPE_LIST = [
    BitType(4, False, 'uint4'),
    BitType(8, True, 'int8'),
    BitType(8, False, 'uint8'),
    BitType(4, True, 'int4'),
    BitType(2, False, 'uint2'),
]
BIT_TYPE_DICT = {bit_type.name: bit_type for bit_type in BIT_TYPE_LIST}


"""
Observer
"""
class BaseObserver:
    def __init__(self, module_type, bit_type, calibration_mode):
        self.module_type = module_type
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.max_val = None
        self.min_val = None
        self.eps = torch.finfo(torch.float32).eps

    def reshape_tensor(self, v):
        if not isinstance(v, torch.Tensor):
            v = torch.tensor(v)
        v = v.detach()
        if self.module_type in ['conv_weight', 'linear_weight']:
            v = v.reshape(v.shape[0], -1)
        elif self.module_type == 'activation':
            if len(v.shape) == 4:
                v = v.permute(0, 2, 3, 1) # [B, H, W, C]
            v = v.reshape(-1, v.shape[-1])
            v = v.transpose(0, 1)
        else:
            raise NotImplementedError
        return v

    def update(self, v):
        # update self.max_val and self.min_val
        raise NotImplementedError

    def get_quantization_params(self, *args, **kwargs):
        raise NotImplementedError

class MinmaxObserver(BaseObserver):
    def __init__(self, module_type, bit_type, calibration_mode):
        super(MinmaxObserver, self).__init__(module_type, bit_type,
                                             calibration_mode)
        self.symmetric = self.bit_type.signed

    def update(self, v):
        v = self.reshape_tensor(v)
        cur_max = v.max(axis=1).values
        # print(f'min_max_weight: {v.shape}')
        # print(f'minmax_cur_max shape: {cur_max.shape}')
        # print(f'minmax_max_val: {self.max_val}')
        
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = torch.max(cur_max, self.max_val)
        cur_min = v.min(axis=1).values
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = torch.min(cur_min, self.min_val)

        if self.calibration_mode == 'layer_wise':
            self.max_val = self.max_val.max()
            self.min_val = self.min_val.min()

    def get_quantization_params(self, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val

        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound
        # print(f'minmax_max_val:{max_val}')
        # print(f'minmax_min_val:{min_val}')

        scale = torch.ones_like(max_val, dtype=torch.float32)
        zero_point = torch.zeros_like(max_val, dtype=torch.int64)

        if self.symmetric:
            max_val = torch.max(-min_val, max_val)
            scale = max_val / (float(qmax - qmin) / 2)
            scale.clamp_(self.eps)
            zero_point = torch.zeros_like(max_val, dtype=torch.int64)
        else:
            scale = (max_val - min_val) / float(qmax - qmin)
            scale.clamp_(self.eps)
            zero_point = qmin - torch.round(min_val / scale)
            zero_point.clamp_(qmin, qmax)
        return scale, zero_point

class PtfObserver(BaseObserver):

    def __init__(self, module_type, bit_type, calibration_mode):
        super(PtfObserver, self).__init__(module_type, bit_type,
                                          calibration_mode)

    def update(self, v): # v = weight
        # print(f'weight_bf_reshape: {v.shape}')
        v = self.reshape_tensor(v)
        
        cur_max = v.max(axis=1).values
        # print(f'weight: {v.shape}')
        # print(f'cur_max shape: {cur_max.shape}')
        # print(f'max_val: {self.max_val.shape if self.max_val is not None else None}')
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = torch.max(cur_max, self.max_val)
            
        cur_min = v.min(axis=1).values
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = torch.min(cur_min, self.min_val)

        if self.calibration_mode == 'layer_wise':
            self.max_val = self.max_val.max()
            self.min_val = self.min_val.min()

    def get_quantization_params(self, inputs, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val

        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound
        # print(f'pt_max_val:{max_val}')
        # print(f'pt_min_val:{min_val}')

        best_score = 1e+10
        max_val_t = max_val.max()
        min_val_t = min_val.min()
        scale8 = (max_val_t - min_val_t) / float(qmax - qmin)
        scale8.clamp_(self.eps)
        scale4 = scale8 / 2
        scale2 = scale4 / 2
        scale1 = scale2 / 2
        zero_point = qmin - torch.round(min_val_t / scale8)
        zero_point.clamp_(qmin, qmax)
        scale_mask = torch.ones_like(max_val)
        # print(f'Input:{inputs.shape}')
        # print(f'scale_mask:{scale_mask.shape}')
        for j in range(inputs.shape[2]):
        # for j in range(scale_mask.shape[0]):
            data = inputs[..., j].unsqueeze(-1)
            data_q1 = ((data / scale1 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale1
            data_q2 = ((data / scale2 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale2
            data_q4 = ((data / scale4 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale4
            data_q8 = ((data / scale8 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale8
            score1 = lp_loss(data, data_q1, p=2.0, reduction='all')
            score2 = lp_loss(data, data_q2, p=2.0, reduction='all')
            score4 = lp_loss(data, data_q4, p=2.0, reduction='all')
            score8 = lp_loss(data, data_q8, p=2.0, reduction='all')
            score = [score1, score2, score4, score8]
            scale_mask[j] *= 2**score.index(min(score))
        scale = scale1 * scale_mask
        return scale, zero_point

"""
Quantizer
"""
# quantizer
class BaseQuantizer(nn.Module):

    def __init__(self, bit_type, observer, module_type):
        super(BaseQuantizer, self).__init__()
        self.bit_type = bit_type
        self.observer = observer
        self.module_type = module_type

    def get_reshape_range(self, inputs):
        range_shape = None
        if self.module_type == 'conv_weight':
            range_shape = (-1, 1, 1, 1)
        elif self.module_type == 'linear_weight':
            range_shape = (-1, 1)
        elif self.module_type == 'activation':
            if len(inputs.shape) == 2:
                range_shape = (1, -1)
            elif len(inputs.shape) == 3:
                range_shape = (1, 1, -1)
            elif len(inputs.shape) == 4:
                range_shape = (1, -1, 1, 1)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return range_shape

    def update_quantization_params(self, *args, **kwargs):
        pass

    def quant(self, inputs, scale=None, zero_point=None):
        raise NotImplementedError

    def dequantize(self, inputs, scale=None, zero_point=None):
        raise NotImplementedError

    def forward(self, inputs):
        outputs = self.quant(inputs)
        outputs = self.dequantize(outputs)
        return outputs

class Log2Quantizer(BaseQuantizer):
    def __init__(self, bit_type, observer, module_type):
        super(Log2Quantizer, self).__init__(
            bit_type,
            observer,
            module_type,
        )
        self.softmax_mask = None

    def quant(self, inputs):
        rounds = torch.round(-1 * inputs.log2())
        self.softmax_mask = rounds >= 2**self.bit_type.bits
        outputs = torch.clamp(rounds, 0, 2**self.bit_type.bits - 1)
        return outputs

    def dequantize(self, inputs):
        outputs = 2**(-1 * inputs)
        outputs[self.softmax_mask] = 0
        return outputs   
    
class UniformQuantizer(BaseQuantizer):
    def __init__(self, bit_type, observer, module_type):
        super(UniformQuantizer, self).__init__(bit_type, observer, module_type)
        self.scale = None # torch.tensor([0, 0, 0]) 
        self.zero_point = None 

    def update_quantization_params(self, *args, **kwargs):
        self.scale, self.zero_point = self.observer.get_quantization_params(
            *args, **kwargs)

    def quant(self, inputs, scale=None, zero_point=None):
        if scale is None:
            scale = self.scale
            # print(f'uniform_scale_bf: {self.scale}')
        if zero_point is None:
            zero_point = self.zero_point
            # print(f'zero-point_bf: {self.zero_point}')
        range_shape = self.get_reshape_range(inputs)
        # print(f"scale: {self.scale.shape}, zero_point: {self.zero_point.shape}")
        # print(f"Input: {inputs.shape}")
        scale = scale.reshape(range_shape)
        zero_point = zero_point.reshape(range_shape)
        outputs = inputs / scale + zero_point
        outputs = outputs.round().clamp(self.bit_type.lower_bound,
                                        self.bit_type.upper_bound)
        return outputs

    def dequantize(self, inputs, scale=None, zero_point=None):
        if scale is None:
            scale = self.scale
        if zero_point is None:
            zero_point = self.zero_point
        range_shape = self.get_reshape_range(inputs)
        scale = scale.reshape(range_shape)
        zero_point = zero_point.reshape(range_shape)
        outputs = (inputs - zero_point) * scale
        return outputs

"""
Quant Module
"""
class QConv2d(nn.Conv2d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 quant=False,
                 calibrate=False,
                 last_calibrate=False,
                 bit_type='int8',
                 calibration_mode='layer_wise',
                 observer_str='minmax',
                 quantizer_str='uniform'):
        super(QConv2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = BIT_TYPE_DICT['int8']
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str
        self.module_type = 'conv_weight'
        if self.observer_str == 'minmax':
            self.observer = MinmaxObserver(self.module_type, self.bit_type, self.calibration_mode)
        elif self.observer_str == 'ptf':
            self.observer = PtfObserver(self.module_type, self.bit_type, self.calibration_mode)
        else:
            raise NotImplementedError
        if self.quantizer_str == 'uniform':
            self.quantizer = UniformQuantizer(self.bit_type, self.observer, self.module_type)
        elif self.quantizer_str == 'log2':
            self.quantizer = Log2Quantizer(self.bit_type, self.observer, self.module_type)
        

    def forward(self, x):
        if self.calibrate:
            # print(f'weight_CONV: {self.weight}')
            self.quantizer.observer.update(self.weight)
            if self.last_calibrate:
                self.quantizer.update_quantization_params(x)
        if not self.quant:
            return F.conv2d(
                x,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        weight = self.quantizer(self.weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)

class QLinear(nn.Linear):

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 quant=False,
                 calibrate=False,
                 last_calibrate=False,
                 bit_type='int8',
                 calibration_mode='layer_wise',
                 observer_str='minmax',
                 quantizer_str='uniform'):
        super(QLinear, self).__init__(in_features, out_features, bias)

        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = BIT_TYPE_DICT[bit_type]
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = 'linear_weight'
        if self.observer_str == 'minmax':
            self.observer = MinmaxObserver(self.module_type, self.bit_type, self.calibration_mode)
        elif self.observer_str == 'ptf':
            self.observer = PtfObserver(self.module_type, self.bit_type, self.calibration_mode)
        else:
            raise NotImplementedError
        if self.quantizer_str == 'uniform':
            self.quantizer = UniformQuantizer(self.bit_type, self.observer, self.module_type)
        elif self.quantizer_str == 'log2':
            self.quantizer = Log2Quantizer(self.bit_type, self.observer, self.module_type)

    def forward(self, x):
        if self.calibrate:
            self.quantizer.observer.update(self.weight)
            if self.last_calibrate:
                self.quantizer.update_quantization_params(x)
        if not self.quant:
            return F.linear(x, self.weight, self.bias)
        weight = self.quantizer(self.weight)
        return F.linear(x, weight, self.bias)

class QAct(nn.Module):

    def __init__(self,
                 quant=False,
                 calibrate=False,
                 last_calibrate=False,
                 bit_type='int8',
                 calibration_mode='layer_wise',
                 observer_str='minmax',
                 quantizer_str='uniform'):
        super(QAct, self).__init__()
        
        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = BIT_TYPE_DICT[bit_type]
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = 'activation'
        if self.observer_str == 'minmax':
            self.observer = MinmaxObserver(self.module_type, self.bit_type, self.calibration_mode)
        elif self.observer_str == 'ptf':
            self.observer = PtfObserver(self.module_type, self.bit_type, self.calibration_mode)
        else:
            print('Error in QAct: observer_str {} is not supported'.format(self.observer_str))
            raise NotImplementedError
        if self.quantizer_str == 'uniform':
            self.quantizer = UniformQuantizer(self.bit_type, self.observer, self.module_type)
        elif self.quantizer_str == 'log2':
            self.quantizer = Log2Quantizer(self.bit_type, self.observer, self.module_type)
        else:
            print('Error in QAct: quantizer_str {} is not supported'.format(self.quantizer_str))
            raise NotImplementedError

    def forward(self, x):
        if self.calibrate:
            # print(f'input_shape: {x.shape}')
            self.quantizer.observer.update(x)
            if self.last_calibrate:
                # import ipdb;ipdb.set_trace()
                self.quantizer.update_quantization_params(x)
                # print("quantizer updated")
        if not self.quant:
            return x
        x = self.quantizer(x)
        return x

class QIntLayerNorm(nn.LayerNorm):

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(QIntLayerNorm, self).__init__(normalized_shape, eps,
                                            elementwise_affine)
        assert isinstance(normalized_shape, int)
        self.mode = 'ln'

    def get_MN(self, x):
        bit = 8
        N = torch.clamp(bit - 1 - torch.floor(torch.log2(x)), 0, 31)
        M = torch.clamp(torch.floor(x * torch.pow(2, N)), 0, 2 ** bit - 1)
        return M, N

    def forward(self,
                x,
                in_quantizer=None,
                out_quantizer=None,
                in_scale_expand=1):
        if self.mode == 'ln':
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias,
                             self.eps)
        elif self.mode == 'int':
            in_scale = in_quantizer.scale
            if in_scale_expand != 1:
                in_scale = in_scale.unsqueeze(-1).expand(
                    -1, in_scale_expand).T.reshape(-1)
            out_scale = out_quantizer.scale
            assert in_scale is not None and out_scale is not None
            channel_nums = x.shape[-1]
            in_scale = in_scale.reshape(1, 1, -1)
            out_scale = out_scale.reshape(1, 1, -1)
            x_q = (x / in_scale).round()
            in_scale1 = in_scale.min()
            in_scale_mask = (in_scale / in_scale1).round()

            x_q = x_q * in_scale_mask

            mean_x_q = x_q.mean(dim=-1) * in_scale1
            std_x_q = (in_scale1 / channel_nums) * torch.sqrt(
                channel_nums * (x_q**2).sum(dim=-1) - x_q.sum(dim=-1)**2)

            A = (in_scale1 / std_x_q).unsqueeze(-1) * \
                self.weight.reshape(1, 1, -1) / out_scale
            A_sign = A.sign()
            M, N = self.get_MN(A.abs())
            B = ((self.bias.reshape(1, 1, -1) -
                  (mean_x_q / std_x_q).unsqueeze(-1) *
                  self.weight.reshape(1, 1, -1)) / out_scale *
                 torch.pow(2, N)).round()

            x_q = ((A_sign * M * x_q + B) / torch.pow(2, N)).round()
            x = x_q * out_scale
        else:
            raise NotImplementedError
        return x

class QIntSoftmax(nn.Module):

    def __init__(self,
                 log_i_softmax=False,
                 quant=False,
                 calibrate=False,
                 last_calibrate=False,
                 bit_type='int8',
                 calibration_mode='layer_wise',
                 observer_str='minmax',
                 quantizer_str='uniform'):
        super(QIntSoftmax, self).__init__()

        self.log_i_softmax = log_i_softmax
        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = BIT_TYPE_DICT[bit_type]
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = 'activation'
        if self.observer_str == 'minmax':
            self.observer = MinmaxObserver(self.module_type, self.bit_type, self.calibration_mode)
        elif self.observer_str == 'ptf':
            self.observer = PtfObserver(self.module_type, self.bit_type, self.calibration_mode)
        else:
            raise NotImplementedError
        if self.quantizer_str == 'uniform':
            self.quantizer = UniformQuantizer(self.bit_type, self.observer, self.module_type)
        elif self.quantizer_str == 'log2':
            self.quantizer = Log2Quantizer(self.bit_type, self.observer, self.module_type)

    @staticmethod
    def log_round(x):
        x_log_floor = x.log2().floor()
        big = x_log_floor
        extra_mask = (x - 2**big) >= 2**(big - 1)
        big[extra_mask] = big[extra_mask] + 1
        return big

    @staticmethod
    def int_softmax(x, scaling_factor):

        def int_polynomial(x_int, scaling_factor):
            coef = [0.35815147, 0.96963238, 1.]  # ax**2 + bx + c
            coef[1] /= coef[0]
            coef[2] /= coef[0]
            b_int = torch.floor(coef[1] / scaling_factor)
            c_int = torch.floor(coef[2] / scaling_factor**2)
            z = x_int + b_int
            z = x_int * z
            z = z + c_int
            scaling_factor = coef[0] * scaling_factor**2
            return z, scaling_factor

        def int_exp(x_int, scaling_factor):
            x0 = -0.6931  # -ln2
            n = 30  # sufficiently large integer
            x0_int = torch.floor(x0 / scaling_factor)
            x_int = torch.max(x_int, n * x0_int)
            q = torch.floor(x_int / x0_int)
            r = x_int - x0_int * q
            exp_int, exp_scaling_factor = int_polynomial(r, scaling_factor)
            exp_int = torch.clamp(torch.floor(exp_int * 2**(n - q)), min=0)
            scaling_factor = exp_scaling_factor / 2**n
            return exp_int, scaling_factor

        x_int = x / scaling_factor
        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int = x_int - x_int_max
        exp_int, exp_scaling_factor = int_exp(x_int, scaling_factor)
        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)
        return exp_int, exp_int_sum

    def forward(self, x, scale):
        if self.log_i_softmax and scale is not None:
            exp_int, exp_int_sum = self.int_softmax(x, scale)
            softmax_out = torch.round(exp_int_sum / exp_int)
            rounds = self.log_round(softmax_out)
            mask = rounds >= 2**self.bit_type.bits
            qlog = torch.clamp(rounds, 0, 2**self.bit_type.bits - 1)
            deq_softmax = 2**(-qlog)
            deq_softmax[mask] = 0
            return deq_softmax
        else:
            x = x.softmax(dim=-1)
            if self.calibrate:
                self.quantizer.observer.update(x)
                if self.last_calibrate:
                    self.quantizer.update_quantization_params(x)
            if not self.quant:
                return x
            x = self.quantizer(x)
            return x

class QMatMul(nn.Module):
    def __init__(self):
        super(QMatMul, self).__init__()
        self.register_buffer('act_scaling_factor', torch.zeros(1))
    
    def fix(self):
        pass

    def unfix(self):
        pass

    def forward(self, A, pre_act_scaling_factor_A, B, pre_act_scaling_factor_B):
        A_int = A / pre_act_scaling_factor_A
        B_int = B / pre_act_scaling_factor_B
        act_scaling_factor = pre_act_scaling_factor_A * pre_act_scaling_factor_B
        self.act_scaling_factor = act_scaling_factor
        return (A_int @ B_int) * act_scaling_factor, act_scaling_factor

"""
Utils
"""
def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (
        x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + \
        torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)






