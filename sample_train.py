import random
import numpy as np
import torch
def Get_data():
    data = torch.randint(10, 100, (100,), dtype=torch.float32)
    data2 = torch.randint(-3, 3, (100,), dtype=torch.float32)
    test_data = data + data2
    return test_data

