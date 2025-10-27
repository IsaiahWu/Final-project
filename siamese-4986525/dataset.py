import random
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
from PIL import Image
import config


class SiameseDataset(Dataset):
  def __init__(self, image_dir, labels_csv, transform=None):
      self.image_dir = config.train_image
      self.labels_df = pd.read_csv(config.train_labels)
      self.transform = transform
      self.create_labels()
      self.pairs = self.pairs_image_label()

  def create_labels(self):
    labels = [] #normal or melonma
    for i in range(len(self.labels_df)):
      if(self.labels_df.loc[i, "diagnosis"] == "melanoma"):
        labels.append(1) #assigned labels to the picture
      else:
        labels.append(0)
    self.labels_df['binary_label'] = labels  # Make it to binary labels


# pairs images with labels
  def pairs_image_label(self):
    image_names = os.listdir(self.image_dir)
    image_dict = {os.path.splitext(name)[0]: name for name in image_names}
    pairs = []
    for _, row in self.labels_df.iterrows():
        image_id = str(row["image_name"]).strip()  # Defensive strip
        label = row["binary_label"]
        if image_id in image_dict:
            full_path = os.path.join(self.image_dir, image_dict[image_id])
            pairs.append((full_path, label))
    print(f"Loaded {len(pairs)} image pairs from {self.image_dir}")
    return pairs



  # How many pairs of images
  def __len__(self):
    return len(self.pairs)


# Preparing labels to comapre
  def __getitem__(self, index):
    img_1, label_1 = self.pairs[index] #sperating image and pair
    same_class = random.randint(0, 1) # Decide if second image should be same class or not
    if same_class:
      while True:
        img_2, label_2 = random.choice(self.pairs) #pick second image randonmly
        if label_1 == label_2:
          pairs_label = 1
          break
    else:
      while True:
        img_2, label_2 = random.choice(self.pairs)
        if label_1 != label_2:
          pairs_label = 0
          break
    img_1 = Image.open(img_1).convert("RGB")
    img_2 = Image.open(img_2).convert("RGB")

    #transfrom image from pixel to number
    if self.transform is not None:
      img_1 = self.transform(img_1)
      img_2 = self.transform(img_2)

    return img_1, img_2, pairs_label
