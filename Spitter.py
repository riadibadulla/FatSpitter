import copy
from torch import nn
import math

class Spitter:
    def __init__(self,model,construction_table, starting_point, number_of_classes):
        self.starting_point = starting_point
        self.original_model = model
        self.construction_table = construction_table
        self.hooks = []
        self.number_of_classes = number_of_classes
        self.i =0
        self.convolutional_iterator = 0
        self.next_input_channels = None

    def replace_layers(self, model):
        for n, module in model.named_children():
            if len(list(module.children())) > 0:
                self.replace_layers(module)
            else:
                if self.i >= self.starting_point:
                    if isinstance(module, nn.AdaptiveAvgPool2d) or isinstance(module, nn.MaxPool2d):
                        new_layer = nn.AdaptiveAvgPool2d(int(math.sqrt(self.number_of_classes)))
                        try:
                            n = int(n)
                            model[n] = new_layer
                        except:
                            setattr(model, n, new_layer)
                    if isinstance(module, nn.Conv2d):
                        new_output_channels = self.construction_table[self.convolutional_iterator][
                                                  "feature_pixels"] // self.number_of_classes
                        new_kernel_size = int(math.sqrt(
                            self.construction_table[self.convolutional_iterator]["number_of_weights"] // (
                                        module.in_channels * new_output_channels)))
                        if not self.next_input_channels:
                            new_layer = nn.Conv2d(module.in_channels,new_output_channels,new_kernel_size,padding="same")
                        else:
                            new_layer = nn.Conv2d(self.next_input_channels, new_output_channels, new_kernel_size, padding="same")
                        try:
                            n = int(n)
                            model[n] = new_layer
                        except:
                            setattr(model, n, new_layer)
                        module = new_layer
                        module.apply(lambda x: x)
                        self.next_input_channels = new_output_channels
                        self.convolutional_iterator += 1
                    if isinstance(module, nn.BatchNorm2d):
                        new_layer = nn.BatchNorm2d(self.next_input_channels)
                        try:
                            n = int(n)
                            model[n] = new_layer
                        except:
                            setattr(model, n, new_layer)
                self.i += 1
        return model


    def get_new_model(self):
        self.fatmodel = copy.deepcopy(self.original_model)
        self.fatmodel = self.replace_layers(self.fatmodel)
        self.fatmodel.zero_grad()
        self.fatmodel._modules["classifier"] = nn.Identity()
        return self.fatmodel

