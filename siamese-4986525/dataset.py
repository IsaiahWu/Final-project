import random
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
from PIL import Image


class Siamese_Datase(Dataset):
  def __init__(self, store_path, train_image, train_labels, transform = None):
    self.train_image = train_image
    self.labels_df = pd.read_csv(train_labels)
    self.transform = transform
    self.store_path = store_path
    self.create_labels()
    self.pairs = self.pair_image_label()

  def create_labels(self):
    labels = [] #normal or melonma
    for i in range(len(self.labels_df)):
      if(self.labels_df.loc[i, "diagnosis"] == "melanoma"):
        labels.append(1) #assigned labels to the picture
      else:
        labels.append(0)
    self.labels_df['binary_label'] = labels  # Make it to binary labels


  def pair_image_label (self):
    image_names = os.listdir(self.train_image)
    pairs = []
    for image_name in image_names:
      match_name = self.labels_df[self.labels_df["image_name"] == image_name]
      if not match_name.empty:
        label = match_name.iloc[0]["binary_label"] # get the label, binary_label
        full_path = os.path.join(self.train_image, image_name) #access to each indivual iamge
        pairs.append((full_path, label))  #("a.jpg", 1)
    return pairs
  
  
  
  # How many pairs of images
  def __len__(self):
    return len(self.pairs)

  def __getitem__(self, index):
    img_1, label_1 = self.pairs[index] #sperating image and pair
    same_class = random.randint(0, 1) # Decide if second image should be same class or not
    if same_class:
      while True:
        img_2, label_2 = random.choice(self.pairs) #pick second image randonmly
        if label_1 == label_2:
          pair_label = 1
          break
    else:
      while True:
        img_2, label_2 = random.choice(self.pairs)
        if label_1 != label_2:
          pair_label = 0
          break
    img_1 = Image.open(img_1).convert("RGB")
    img_2 = Image.open(img_2).convert("RGB")

    #transfrom image from pixel to number
    if self.transform is not None:
      img_1 = self.transform(img_1)
      img_2 = self.transform(img_2)

    return img_1, img_2, pair_label









