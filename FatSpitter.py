import torch

from Spitter import Spitter
import Construction_table
from Construction_table import Construction_table
from torchsummary import summary
from torch import nn
from optnn import OpticalConv2d
#models
from models import ResNet
from models import contracting_UNet

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

def fat_spitter(model,input_size, number_of_classes, optical=False):
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
    construction_table = Construction_table(model,number_of_classes)
    table, starting_point = construction_table(model, input_size=input_size)
    for layer in table:
        print(layer)
    spitter = Spitter(model,table,starting_point, number_of_classes, optical=optical)
    new_model = spitter()
    new_model.apply(lambda x: x)
    return new_model



if __name__ == '__main__':
    model = contracting_UNet().to(torch.device("cuda"))
    # model = ResNet().to(torch.device("cpu"))
    fat_model = fat_spitter(model,input_size=(3,32,32), number_of_classes=100, optical=False)
    print(fat_model.cuda())
    summary(fat_model, (3, 32, 32), device="cuda")
    print(get_number_of_conv_operations(model))
    print(get_number_of_conv_operations(fat_model))