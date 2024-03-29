import copy
from torch import nn
import math
from optnn import OpticalConv2d

class Spitter:
    """
    A class which is resposible of the FatNet conversion.

    :param model: model to be converted
    :type nn.Module
    :param construction_table: consturuction table achieved via Construction_table class
    :type list of dictionaries
    :param starting_point: the index of the layer to start the conversion from
    :type int
    :param number_of_classes: number of classes of the dataset
    :type int
    :param optical: if True use OpticalConv2d
    :type bool
    """
    def __init__(self,model,construction_table, starting_point, number_of_classes, optical=False):
        self.starting_point = starting_point
        self.original_model = model
        self.construction_table = construction_table
        self.hooks = []
        self.number_of_classes = number_of_classes
        self.i =0
        self.index_of_previous_convolutions = 0
        self.convolutional_iterator = 0
        self.next_input_channels = None
        self.is_optical = optical
        self.reference_to_previous_layer = None
        self.input_channels_of_previous_layer = None
        self.first_refined_layer = True
        self.refinement_dict = {}
        self.refinement_index_to_convid = {}
        # self.index_of_conv_needs_replacement = None
        # self.output_channels_of_conv_tobereplaced = None

    def _replace_the_layer(self,model,n,new_layer):
        """
        Given the name or index of the layer, and new layer, replace the old layer with new.
        :param model: entire model
        :param n: index or name of layer in the model, which needs to be raplaced
        :type int/str
        :param new_layer: new layer replacement layer
        :type nn.Module
        :return: new model with replaced layer
        :return: nn.Module
        """
        try:
            n = int(n)
            model[n] = new_layer
        except:
            setattr(model, n, new_layer)
        return model

    def replace_layers(self, model):
        """
        Recursively converts the model into FatNet based on the construction table
        :param model: original mode
        :type nn.Module
        :return: FatNet module
        :rtype: nn.Module
        """
        for n, module in model.named_children():
            if len(list(module.children())) > 0:
                self.replace_layers(module)
            else:
                if self.i >= self.starting_point:
                    if isinstance(module, nn.AdaptiveAvgPool2d) or isinstance(module, nn.MaxPool2d):
                        new_layer = nn.AdaptiveAvgPool2d(int(math.sqrt(self.number_of_classes)))
                        model = self._replace_the_layer(model,n,new_layer)
                    if isinstance(module, nn.Conv2d):
                        #new output channelels based on new number of pixels
                        if not self.next_input_channels:
                            new_input_channels = module.in_channels
                            self.next_input_channels = new_input_channels
                        else:
                            new_input_channels = self.next_input_channels
                        new_output_channels = int(math.ceil(self.construction_table[self.convolutional_iterator][
                                                                "feature_pixels"] / self.number_of_classes))
                        new_kernel_size = int(math.ceil(math.sqrt(
                            self.construction_table[self.convolutional_iterator]["number_of_weights"] // (
                                    new_input_channels * new_output_channels))))
                        if module.in_channels == module.out_channels:
                            if new_output_channels!=new_input_channels or new_kernel_size**2>self.number_of_classes:
                                new_kernel_size = min(new_kernel_size, int(math.sqrt(self.number_of_classes)))
                                #we devide by kernel size, unless its bigger than
                                new_channels = int(math.ceil(math.sqrt(self.construction_table[self.convolutional_iterator][
                                                                               "number_of_weights"] / new_kernel_size**2)))
                                new_input_channels, new_output_channels = new_channels, new_channels
                        else:
                            if new_kernel_size**2>self.number_of_classes:
                                new_kernel_size = int(math.ceil(math.sqrt(self.number_of_classes)))
                                new_output_channels = int(
                                    self.construction_table[self.convolutional_iterator]["number_of_weights"] // (
                                                new_input_channels * new_kernel_size ** 2))
                            #and kernel is clipped get new out channels
                        if self.is_optical:
                            new_layer = OpticalConv2d(new_input_channels,new_output_channels,new_kernel_size,True,True,input_size=int(math.sqrt(self.number_of_classes)))
                        else:
                            new_layer = nn.Conv2d(new_input_channels,new_output_channels,new_kernel_size,padding="same")

                        #if module.in_channels != new_in_channels the index of previous layer in the dictionary
                        # if self.next_input_channels != new_input_channels and self.next_input_channels:
                        if self.next_input_channels != new_input_channels:
                            print(module)
                            print(self.i)
                            print(self.convolutional_iterator)
                            print(self.index_of_previous_convolutions)
                            self.refinement_dict[self.convolutional_iterator - 1] = new_output_channels
                            self.refinement_index_to_convid[self.index_of_previous_convolutions] = self.convolutional_iterator - 1
                        model = self._replace_the_layer(model,n,new_layer)
                        self.next_input_channels = new_output_channels
                        self.convolutional_iterator += 1
                    if isinstance(module, nn.BatchNorm2d):
                        new_layer = nn.BatchNorm2d(self.next_input_channels)
                        model = self._replace_the_layer(model,n,new_layer)
                    if isinstance(module, nn.Flatten):
                        new_layer = nn.Identity()
                        model = self._replace_the_layer(model,n,new_layer)
                    if isinstance(module, nn.Linear):
                        new_layer = nn.Conv2d(in_channels=self.next_input_channels,out_channels=1,kernel_size=int(math.sqrt(self.number_of_classes)),padding="same")
                        model = self._replace_the_layer(model, n, new_layer)
                        self.next_input_channels = 1
                if isinstance(module, nn.Conv2d):
                    self.index_of_previous_convolutions = self.i
                self.i += 1
        return model

    def refine_on_of_the_layers(self, model):
        """
        Recursively converts the model into FatNet based on the construction table
        :param model: original mode
        :type nn.Module
        :return: FatNet module
        :rtype: nn.Module
        """
        for n, module in model.named_children():
            if len(list(module.children())) > 0:
                self.refine_on_of_the_layers(module)
            else:
                if self.i >= min(self.refinement_index_to_convid.keys()):
                    if isinstance(module, nn.Conv2d):
                        if self.convolutional_iterator in self.refinement_dict.keys():
                            out_c = self.refinement_dict[self.convolutional_iterator]
                            input_c = module.in_channels
                            if self.convolutional_iterator != -1:
                                weights = self.construction_table[self.convolutional_iterator]["number_of_weights"]
                            else:
                                print(module.kernel_size)
                                weights = module.in_channels*module.out_channels*module.kernel_size[0]**2
                            kernel_size = int(math.ceil(math.sqrt(weights/(input_c*out_c))))
                            if self.is_optical:
                                new_layer = OpticalConv2d(input_c, out_c, kernel_size, True,
                                                          True, input_size=int(math.sqrt(self.number_of_classes)))
                            else:
                                new_layer = nn.Conv2d(input_c,out_c,kernel_size,padding="same")
                            model = self._replace_the_layer(model, n, new_layer)
                            self.next_input_channels = out_c
                        self.convolutional_iterator += 1
                    if self.convolutional_iterator-1 in self.refinement_dict.keys() and isinstance(module, nn.BatchNorm2d):
                        new_layer = nn.BatchNorm2d(self.next_input_channels)
                        model = self._replace_the_layer(model,n,new_layer)
                self.i += 1
        return model

    def add_flatten(self, model):
        class new_model(nn.Module):
            def __init__(self,model):
                super().__init__()
                self.one_layer = nn.Sequential(model, nn.Flatten())

            def forward(self,x):
                x = self.one_layer(x)
                return x

        return new_model(model)

    def __call__(self):
        """
        Perform the conversion
        :return: Fatnet model
        :rtype nn.Module
        """
        self.fatmodel = copy.deepcopy(self.original_model)
        self.fatmodel = self.replace_layers(self.fatmodel)
        if bool(self.refinement_dict):
            self.convolutional_iterator = min(self.refinement_dict.keys())
            self.i = 0
            print(self.fatmodel)
            print(self.refinement_dict)
            print(self.refinement_index_to_convid)
            self.fatmodel = self.refine_on_of_the_layers(self.fatmodel)
        self.fatmodel.zero_grad()
        #Need to add another layer of flatten to make the network trainable for classification
        self.fatmodel = self.add_flatten(self.fatmodel)
        return self.fatmodel

