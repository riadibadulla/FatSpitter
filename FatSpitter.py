import torch

from Spitter import Spitter
import Construction_table
from Construction_table import Construction_table
from models import ResNet
from collections import OrderedDict
from torchsummary import summary
import numpy as np
model = ResNet()
construction_table = Construction_table(model,100)
table, starting_point = construction_table.get_table(model, input_size=(3,32,32))
for layer in table:
    print(layer)
spitter = Spitter(model,table,starting_point, 100)
new_model = spitter.get_new_model()
new_model.apply(lambda x: x)
new_model(torch.rand([2,3,32,32]))
summary(new_model,(3,32,32))
summary(new_model,(3,32,32))
