from models import ResNet
import math
import torch
import torch.nn as nn


model = None
number_of_classes = None
construction_table =[]

idx = 0
last_conv_layer = 0
stop_checking = False

def get_table(model, input_size, batch_size=-1, device="cuda"):
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

    def get_construction_table(module):
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
                    if not construction_table:
                        number_of_weights = torch.prod(torch.LongTensor(list(module.weight.size())))
                        construction_table.append({"number_of_weights":number_of_weights,"feature_pixels":0})
                    else:
                        number_of_weights = torch.prod(torch.LongTensor(list(module.weight.size())))
                        feature_pixels_of_previous_convolution = list(input[0].size())[1]*list(input[0].size())[2]*list(input[0].size())[3]
                        construction_table[-1]["feature_pixels"] = feature_pixels_of_previous_convolution
                        construction_table.append({"number_of_weights": number_of_weights, "feature_pixels": 0})
                    # pixels = output_shape[3] * output_shape[1] * output_shape[2]
                    # new_out_channels = pixels // number_of_classes
                    # old_parameters = torch.prod(torch.LongTensor(list(module.weight.size())))
                    # new_kernel = math.sqrt(old_parameters/(list(input[0].size())[1]*new_out_channels))
                    # if not new_conv_layers:
                    #     new_conv_layers.append({"input_channels": list(input[0].size())[1], "output_channels":new_out_channels, "kernel_size":new_kernel})
                    # else:
                    #     new_input_channels = new_conv_layers[-1]["output_channels"]
                    #     new_conv_layers.append({"input_channels": new_input_channels, "output_channels":new_out_channels, "kernel_size":new_kernel})
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
    model.apply(get_construction_table)
    model(*x)
    # remove these hooks
    for h in hooks:
        h.remove()

    construction_table[-1]["feature_pixels"] = construction_table[-2]["feature_pixels"]
    return construction_table

def set_model(model_name,classes):
    global model, number_of_classes
    model = model_name
    number_of_classes = classes
