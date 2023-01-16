from models import ResNet
import math
import torch
import torch.nn as nn

device = "cuda"
device = device.lower()
assert device in [
    "cuda",
    "cpu",
], "Input device is not valid, please specify 'cuda' or 'cpu'"

if device == "cuda" and torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

class Construction_table:

    def __init__(self, model, number_of_classes):
        self.model = model
        self.number_of_classes = number_of_classes
        self.construction_table =[]
        self.idx = 0
        self.last_conv_layer = 0
        self.stop_checking = False
        self.hooks = []

    def get_first_idx(self,module):
        self.idx = 0

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            if isinstance(output, (list, tuple)):
                output_shape = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                output_shape = list(output.size())
                #batch size
                output_shape[0] = -1
            if class_name == "Conv2d" and (not self.stop_checking):
                self.last_conv_layer = self.idx
            try:
                self.stop_checking = output_shape[2] <= math.sqrt(self.number_of_classes)
            except:
                pass
            self.idx += 1

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == self.model)
        ):
            self.hooks.append(module.register_forward_hook(hook))

    def get_construction_table(self, module):
        self.idx = 0

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            if isinstance(output, (list, tuple)):
                output_shape = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                output_shape = list(output.size())
                #batch size
                output_shape[0] = -1
            if class_name == "Conv2d":
                if self.idx >= self.last_conv_layer:
                    if not self.construction_table:
                        number_of_weights = torch.prod(torch.LongTensor(list(module.weight.size()))).item()
                        self.construction_table.append({"number_of_weights": number_of_weights, "feature_pixels": 0})
                    else:
                        number_of_weights = torch.prod(torch.LongTensor(list(module.weight.size()))).item()
                        feature_pixels_of_previous_convolution = list(input[0].size())[1] * list(input[0].size())[2] * \
                                                                 list(input[0].size())[3]
                        self.construction_table[-1]["feature_pixels"] = feature_pixels_of_previous_convolution
                        self.construction_table.append({"number_of_weights": number_of_weights, "feature_pixels": 0})
            self.idx += 1

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == self.model)
        ):
            self.hooks.append(module.register_forward_hook(hook))

    def get_table(self,model, input_size, batch_size=-1):
        # multiple inputs to the network
        if isinstance(input_size, tuple):
            input_size = [input_size]

        # batch_size of 2 for batchnorm
        x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]

        # get_first_idx
        model.apply(self.get_first_idx)
        model(*x)
        # remove these hooks
        for h in self.hooks:
            h.remove()

        # register hook
        model.apply(self.get_construction_table)
        model(*x)
        # remove these hooks
        for h in self.hooks:
            h.remove()

        self.construction_table[-1]["feature_pixels"] = self.construction_table[-2]["feature_pixels"]
        return self.construction_table


