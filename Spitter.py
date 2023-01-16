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

    def recursive_editor(self, model):
        for name, child in model.named_children():
            if isinstance(child, nn.Sequential):
                self.recursive_editor(child)
            else:
                if self.i >= self.starting_point:
                    if isinstance(child, nn.AdaptiveAvgPool2d):
                        # child = nn.AdaptiveAvgPool2d(math.sqrt(self.number_of_classes))
                        #TODO: need to check for all types of pooling and replace them fully
                        child.output_size = int(math.sqrt(self.number_of_classes))
                        print("replaced")
                    if isinstance(child, nn.Conv2d):
                        new_output_channels = self.construction_table[self.convolutional_iterator]["feature_pixels"] // self.number_of_classes
                        new_kernel_size = int(math.sqrt(self.construction_table[self.convolutional_iterator]["number_of_weights"]//(child.in_channels * new_output_channels)))
                        if not self.next_input_channels:
                            # new_layer = nn.Conv2d(child.in_channels,new_output_channels,new_kernel_size,padding="same")
                            child.out_channels=new_output_channels
                            child.kernel_size=new_kernel_size
                        else:
                            # new_layer = nn.Conv2d(self.next_input_channels, new_output_channels, new_kernel_size, padding="same")
                            #TODO: Need to make the replacement in the future for optical layers
                            child.out_channels = new_output_channels
                            child.kernel_size = new_kernel_size
                            child.in_channels=self.next_input_channels
                        # setattr(child,name,new_layer)
                        # child = new_layer
                        self.next_input_channels = new_output_channels
                        self.convolutional_iterator+=1
                self.i += 1
        return model



    def get_new_model(self):
        fatmodel = copy.deepcopy(self.original_model)
        fatmodel = self.recursive_editor(fatmodel)
        fatmodel._modules["classifier"] = nn.Identity()
        return fatmodel

