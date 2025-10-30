# Project Title
Siamese Network Classifier for Skin Lesion Images

## Project Description
Siamese network is a neural network that is currently widely used in facial recognition. [1] The network trains by creating pairs, labeling them as similar or dissimilar. Then each image will be processed through convolution layers outputting embedding vectors. Capturing the patterns and the key features of this image. The machine will compare between embedding vectors using distance matrices and determine the contrastive loss. If the images are similar, the loss is small, if different the loss will be large. Through training, the network learns to minimize the contrastive loss with similar pairs and to improve its ability to identify similar and different images effectively. 

This project will be implementing a siamese network aiming to differentiate between benign skin and melanoma skin cancer through learning “The ISIC 2020 Challenge Dataset”. The goal is to achieve an 80% accuracy by classifying skin images either being benign or melanoma.

## Dataset

This project uses the **ISIC 2020 Challenge Dataset**.

### Files to Download

- **Training Images:** JPEG format (~23GB)  
- **Training Labels:** ISIC_2020_Training_GroundTruth.csv (~2MB), contains image names, diagnosis, and patient metadata  
- **Test Images:** JPEG format (~6.7GB)  
- **Test Labels:** ISIC_2020_Test_Metadata.csv (~458KB)

> **Note:** Please do **not** download the DICOM files as they require special software to process and don't need for this project

### Dataset Access
The dataset is available here: [ISIC 2020 Challenge Dataset](https://challenge2020.isic-archive.com/)

## Data Transformation

This project applies two types of data transformations to the images:

- **Preprocessing**:  
  Images are resized to a fixed size (e.g., 128x128 pixels) and normalized to standardize pixel values for consistent input to the network.

- **Augmentation**:  
  To balance the dataset additional random augmentations such as horizontal flips, rotations, and color change are applied only to melanoma images that are oversampled.



## Model

## How it works

## Dependencies

## How to installation

## Lost function

## Prediction 

## Result

## Future improvement

