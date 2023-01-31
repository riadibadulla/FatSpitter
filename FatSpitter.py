import torch

from Spitter import Spitter
import Construction_table
from Construction_table import Construction_table
from models import ResNet
from torchsummary import summary
from torch import nn
from optnn import OpticalConv2d

def get_number_of_conv_operations(model):
    """
    Get number of convolution operations of the given model
    :param model:PyTorch model
    :type nn.Module
    :return:number of convolution operations in all layers
    :rtype int
    """
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
    """
    Turns any convolutional network into FatNet. Accepts any pytorch model object and spits out the FatNet model
    :param model: Pytorch's network object.
    :type nn.Module
    :param optical: will use OpticalConv2d if True, Conv2d if False
    :type bool
    :return FatNet equivalent of the original model
    :rtype nn.Module
    """
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
    summary(fat_model, (3, 32, 32), device="cpu")
    print(get_number_of_conv_operations(model))
    print(get_number_of_conv_operations(fat_model))