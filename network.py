from brevitas.quant import IntBias
from brevitas.inject.enum import ScalingImplType
from brevitas.inject.defaults import Int8ActPerTensorFloatMinMaxInit
import brevitas.nn as qnn
from torch import nn
import torch

## Making and loading the model


class MSCLayer(nn.Module):
    def __init__(self, output_filters, w_bits, a_bits, bias=True):
        super(MSCLayer, self).__init__()
        filters_in = 2
        stride = 4
        ksize1 = 32
        padding1 = ksize1 // 2
        filter_num1 = output_filters // 3
        self.branch1 = nn.Sequential(
            qnn.QuantConv1d(filters_in, filter_num1, ksize1, padding=padding1, weight_bit_width=w_bits, bias=bias, stride=stride),
            nn.BatchNorm1d(filter_num1),
            qnn.QuantReLU(bit_width=a_bits),
            qnn.QuantConv1d(filter_num1, filter_num1, 1, stride=1, padding=0, weight_bit_width=w_bits, bias=bias),
            nn.BatchNorm1d(filter_num1)
        )

        ksize2 = 26
        padding2 = ksize2 // 2
        filter_num2 = output_filters // 3

        self.branch2 = nn.Sequential(
            qnn.QuantConv1d(filters_in, filter_num2, ksize2, padding=padding2, weight_bit_width=w_bits, bias=bias, stride=stride),
            nn.BatchNorm1d(filter_num2),
            qnn.QuantReLU(bit_width=a_bits),
            qnn.QuantConv1d(filter_num2, filter_num2, 1, stride=1, padding=0, weight_bit_width=w_bits, bias=bias),
            nn.BatchNorm1d(filter_num2)
        )

        ksize3 = 18
        padding3 = ksize3 // 2
        filter_num3 = output_filters // 3

        self.branch3 = nn.Sequential(
            qnn.QuantConv1d(filters_in, filter_num3, ksize3, padding=padding3, weight_bit_width=w_bits, bias=bias, stride=stride),
            nn.BatchNorm1d(filter_num3),
            qnn.QuantReLU(bit_width=a_bits),
            qnn.QuantConv1d(filter_num3, filter_num3, 1, stride=1, padding=0, weight_bit_width=w_bits, bias=bias),
            nn.BatchNorm1d(filter_num3)
        )

    def _forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)

        outputs = [branch1, branch2, branch3]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

class AaronBlock(nn.Module):
    def __init__(self, filters_conv_in, filters_conv_out, ksize, padding, w_bits, a_bits, pool_size=2, bias=True):
        super(AaronBlock, self).__init__()
        self.block = nn.Sequential(
          qnn.QuantConv1d(filters_conv_in, filters_conv_in, ksize, padding=padding, groups=filters_conv_in, weight_bit_width=w_bits, bias=bias),
          nn.BatchNorm1d(filters_conv_in),
          qnn.QuantReLU(bit_width=a_bits),
          qnn.QuantConv1d(filters_conv_in, filters_conv_out, 1, stride=1, padding=0, weight_bit_width=w_bits, bias=bias),
          nn.BatchNorm1d(filters_conv_out),
          nn.MaxPool1d(pool_size)
        )
    def forward(self, x):
        x = self.block(x)
        return x

class AaronNet(nn.Module):
    def __init__(self, config, bits):
        super(AaronNet, self).__init__()
        
        class InputQuantizer(Int8ActPerTensorFloatMinMaxInit):
            bit_width = bits
            min_val = -2.0
            max_val = 2.0
            scaling_impl_type = ScalingImplType.CONST # Fix the quantization range to [min_val, max_val]

        self.input_ = qnn.QuantHardTanh(act_quant=InputQuantizer)
        conf = config["conv_layers"]
        conf0, conf1 = conf[0], conf[1:]
        filters_conv0, ksize0, padding0, w_bits0, a_bits0, pool_size0, bias0 = conf0

        layers = []
        self.expand = MSCLayer(filters_conv0, w_bits0, a_bits0)
        filters_in = filters_conv0
    
        for filters, k, pad, w_bit, a_bit, pool, bias in conf1:
            layers.append(AaronBlock(filters_in, filters, k, pad, w_bit, a_bit, pool, bias))
            filters_in = filters
        self.feats = nn.Sequential(*layers)
        self.pooling = nn.AdaptiveMaxPool1d(1)
        filters_dense, w_bits, a_bits, bias = config["linear_layers"]

        self.classifier = nn.Sequential(
            qnn.QuantLinear(filters_in, filters_dense, weight_bit_width=w_bits, bias=bias),
            nn.BatchNorm1d(filters_dense),
            qnn.QuantReLU(bit_width=a_bits),

            qnn.QuantLinear(filters_dense, filters_dense, weight_bit_width=w_bits, bias=bias),
            nn.BatchNorm1d(filters_dense),
            qnn.QuantReLU(bit_width=a_bits, return_quant_tensor=True),

            qnn.QuantLinear(filters_dense, 24, weight_bit_width=w_bits, bias=True, bias_quant=IntBias),
        )
    def forward(self, x):
        x = self.input_(x)
        x = self.expand(x)
        x = self.feats(x)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



def get_baseline_network(input_bits=8, a_bits=8, w_bits=8, filters_conv=64, filters_dense=128):
    class InputQuantizer(Int8ActPerTensorFloatMinMaxInit):
        bit_width = input_bits
        min_val = -2.0
        max_val = 2.0
        scaling_impl_type = ScalingImplType.CONST
    
    model_vgg = nn.Sequential(
        qnn.QuantHardTanh(act_quant=InputQuantizer),

        qnn.QuantConv1d(2, filters_conv, 3, padding=1, weight_bit_width=w_bits, bias=False),
        nn.BatchNorm1d(filters_conv),
        qnn.QuantReLU(bit_width=a_bits),
        nn.MaxPool1d(2),

        qnn.QuantConv1d(filters_conv, filters_conv, 3, padding=1, weight_bit_width=w_bits, bias=False),
        nn.BatchNorm1d(filters_conv),
        qnn.QuantReLU(bit_width=a_bits),
        nn.MaxPool1d(2),

        qnn.QuantConv1d(filters_conv, filters_conv, 3, padding=1, weight_bit_width=w_bits, bias=False),
        nn.BatchNorm1d(filters_conv),
        qnn.QuantReLU(bit_width=a_bits),
        nn.MaxPool1d(2),

        qnn.QuantConv1d(filters_conv, filters_conv, 3, padding=1, weight_bit_width=w_bits,bias=False),
        nn.BatchNorm1d(filters_conv),
        qnn.QuantReLU(bit_width=a_bits),
        nn.MaxPool1d(2),

        qnn.QuantConv1d(filters_conv, filters_conv, 3, padding=1, weight_bit_width=w_bits, bias=False),
        nn.BatchNorm1d(filters_conv),
        qnn.QuantReLU(bit_width=a_bits),
        nn.MaxPool1d(2),

        qnn.QuantConv1d(filters_conv, filters_conv, 3, padding=1, weight_bit_width=w_bits, bias=False),
        nn.BatchNorm1d(filters_conv),
        qnn.QuantReLU(bit_width=a_bits),
        nn.MaxPool1d(2),

        qnn.QuantConv1d(filters_conv, filters_conv, 3, padding=1, weight_bit_width=w_bits, bias=False),
        nn.BatchNorm1d(filters_conv),
        qnn.QuantReLU(bit_width=a_bits),
        nn.MaxPool1d(2),

        nn.Flatten(),

        qnn.QuantLinear(filters_conv*8, filters_dense, weight_bit_width=w_bits, bias=False),
        nn.BatchNorm1d(filters_dense),
        qnn.QuantReLU(bit_width=a_bits),

        qnn.QuantLinear(filters_dense, filters_dense, weight_bit_width=w_bits, bias=False),
        nn.BatchNorm1d(filters_dense),
        qnn.QuantReLU(bit_width=a_bits, return_quant_tensor=True),

        qnn.QuantLinear(filters_dense, 24, weight_bit_width=w_bits, bias=True, bias_quant=IntBias),
    )

    return model_vgg