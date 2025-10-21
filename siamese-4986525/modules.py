import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseNetwork(nn.Module):
  def __init__(self):
    super(SiameseNetwork, self).__init__()
    self.conv1 = nn.Conv2d (in_channels = 3, out_channels = 16, kernel_size = 3) # Defining my fliter
    
