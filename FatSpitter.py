import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict
import numpy as np
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import Construction_table
from Construction_table import Construction_table
from models import ResNet
from collections import OrderedDict
import numpy as np
model = ResNet()
construction_table = Construction_table(model,100)
table = construction_table.get_table(model, input_size=(3,32,32))
for layer in table:
    print(layer)
