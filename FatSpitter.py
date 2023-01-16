from models import ResNet
import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict
import numpy as np
from torchsummary import summary
import math
import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np


model = ResNet()
number_of_classes = 100
new_conv_layers =[]

idx = 0
last_conv_layer = 0
stop_checking = False

def summary(model, input_size, batch_size=-1, device="cuda"):
    global idx, last_conv_layer,stop_checking

    def get_first_idx(module):
        global idx
        idx = 0
        def hook(module, input, output):
            global stop_checking, last_conv_layer, idx
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            if isinstance(output, (list, tuple)):
                output_shape = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                output_shape = list(output.size())
                output_shape[0] = batch_size
            if class_name == "Conv2d" and (not stop_checking):
                last_conv_layer = idx
            try:
                if output_shape[2] <= math.sqrt(number_of_classes):
                    stop_checking = True
            except:
                pass
            idx += 1
        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    def register_hook_get_fatnet(module):
        global idx
        idx = 0
        def hook(module, input, output):
            global stop_checking, last_conv_layer, idx
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            if isinstance(output, (list, tuple)):
                output_shape = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                output_shape = list(output.size())
                output_shape[0] = batch_size
            if class_name == "Conv2d":
                if idx>=last_conv_layer:
                    pixels = output_shape[3] * output_shape[1] * output_shape[2]
                    new_out_channels = pixels // number_of_classes
                    old_parameters = torch.prod(torch.LongTensor(list(module.weight.size())))
                    new_kernel = math.sqrt(old_parameters/(list(input[0].size())[1]*new_out_channels))
                    if not new_conv_layers:
                        new_conv_layers.append({"input_channels": list(input[0].size())[1], "output_channels":new_out_channels, "kernel_size":new_kernel})
                    else:
                        new_input_channels = new_conv_layers[-1]["output_channels"]
                        new_conv_layers.append({"input_channels": new_input_channels, "output_channels":new_out_channels, "kernel_size":new_kernel})
            idx += 1

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    hooks = []

    # get_first_idx
    model.apply(get_first_idx)
    model(*x)
    # remove these hooks
    for h in hooks:
        h.remove()

    # register hook
    model.apply(register_hook_get_fatnet)
    model(*x)
    # remove these hooks
    for h in hooks:
        h.remove()



summary(model, input_size=(3,32,32))
for layer in new_conv_layers:
    print(layer)
