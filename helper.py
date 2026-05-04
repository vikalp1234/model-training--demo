import os
import json
import torch    
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tensorflow as tf
from collections import defaultdict

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: {left.shape} vs {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))