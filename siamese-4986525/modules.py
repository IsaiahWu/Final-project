import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
  def __init__(self):
    super(SiameseNetwork, self).__init__()
    self.conv1 = nn.Conv2d (in_channels = 3, out_channels = 16, kernel_size = 3, padding = 1) # Defining my fliter
    self.relu = nn.ReLU () #Let all negative outputpatch adjust to 0
    self.pool_1 = nn.MaxPool2d (kernal_size = 2, stride = 2) #Maxpooling, retriving the largest input value 
    self.conv2 = nn.Conv2d (in_channels = 16, out_channels = 32, kernel_size = 3) 
    self.relu = nn.ReLU()
    self.pool_2 = nn.MaxPool2d (kernel_size = 2, stride = 2)
    self.flatten = nn.Flatten()

# How my model flow
  def forward(self, x):
    x = self.conv1(x)
    x = self.relu(x)
    x = self.pool_1(x)
    x = self.conv2(x)
    x = self.relu(x)
    x = self.pool_2(x)
    x = self.flatten(x)
    return x
  
