import torch

from Spitter import Spitter
import Construction_table
from Construction_table import Construction_table
from models import ResNet
from collections import OrderedDict
from torchsummary import summary
import numpy as np
from torch import nn
from optnn import OpticalConv2d

def get_number_of_conv_operations(model):
    conv_opps_sumer = 0
    def multiply_channels(module):
        nonlocal conv_opps_sumer
        if isinstance(module, nn.Conv2d):
            conv_opps_sumer += module.in_channels*module.out_channels
        if isinstance(module,OpticalConv2d):
            conv_opps_sumer += module.input_channels * module.output_channels
    model.apply(multiply_channels)

    return conv_opps_sumer

def fat_spitter(model,optical=False):
    device = torch.device("cpu")
    model = model.to(device)
    construction_table = Construction_table(model,100)
    table, starting_point = construction_table(model, input_size=(3,32,32))
    for layer in table:
        print(layer)
    spitter = Spitter(model,table,starting_point, 100, optical=optical)
    new_model = spitter()
    new_model.apply(lambda x: x)
    return new_model



if __name__ == '__main__':
    model = ResNet().to(torch.device("cpu"))
    fat_model = fat_spitter(model,optical=False)
    print(fat_model)
    # summary(fat_model, (3, 32, 32), device="cpu")
    print(get_number_of_conv_operations(model))
    print(get_number_of_conv_operations(fat_model))