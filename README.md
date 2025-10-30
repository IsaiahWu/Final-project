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

### Data Transformation

This project applies two types of data transformations to the images:

- **Preprocessing**:  
  Images are resized to a fixed size (e.g., 128x128 pixels) and normalized to standardize pixel values for consistent input to the network.

- **Augmentation**:  
  The ISIC 2020 Challenge Dataset contains 33,126 dermoscopic images, with a significant class imbalance. Approximately 26,033 benign images and only 467 melanoma images indicate a ratio of about 1 to 56. This extremely high imbalance will cause overfitting to the majority class, making it difficult to identify the minority class melanoma. To address this issue, data augmentation is needed to apply to the minority classes melanoma images. These augmentation will preserve the essential features of the image and but change the weights create new augmented images. 

## Data Directory Configuration
After downloading the entire dataset to your local machine, follow these steps:

- If you are running the project on your local machine, you can skip the upload step
- If running on a cloud server like [Vast.ai](http://vast.ai), upload the dataset files to your cloud drive

Next, create a configuration file (e.g., `config.py`) to specify the directory paths for your dataset files like this:
```
train_image = 'path/to/train/images'
train_labels = 'path/to/train/labels'
```
Make sure to update these paths according to where the data is stored on your server


### Environment Setup and GPU Requirements

This project was developed and tested using a Vast.ai GPU server with an NVIDIA RTX 5090 (CUDA 13.0) and PyTorch running with CUDA 12.9.1+.

#### Minimum System Requirements
- **GPU:** NVIDIA GPU with CUDA capability (version 11.8 or higher recommended)  
- **CUDA Toolkit:** Version 11.8 or newer  
- **RAM:** At least 32GB (recommended for handling large image datasets)  
- **Storage:** Minimum 30GB free disk space for ISIC dataset and outputs  
- **Python:** Version 3.8+  
- **Frameworks:** PyTorch 2.1+, torchvision 0.16+, along with other compatible packages  

---

### PyTorch Installation

On [Vast.ai](http://vast.ai), PyTorch is pre-installed already in the environment 

If you are using a local machine or another server without PyTorch pre-installed, please follow the official installation guide:  
[PyTorch Installation Guide](https://pytorch.org/get-started/locally/)

---

### Package Installation

Install essential Python packages with pip:
```
pip install pandas pillow numpy matplotlib scikit-learn
```

### Package Versions Used

- PyTorch version: 2.8.0+cu128  
- Torchvision version: 0.23.0+cu128  
- Pandas version: 2.3.3  
- Pillow version: 11.0.0  
- NumPy version: 2.1.2  
- Matplotlib version: 3.10.7  
- Scikit-learn version: 1.7.2  


### Dataset Module

Create a directory for data. This data will handle all the loading and preparing data to be passed in the model. 

## Dataset module structure 

**1. Loading Dataset:**
  Create a constructor class (`__init__`).  This class is responsible for preparing all the dataset whenever it is requested. Specifying the directory and setting up transformation and augmentation
  
**2. Create labels:**
  Convert all the diagnosis classes in “ISIC_2020_Training_GroundTruth.csv” into binary labels melanoma being 1 and benign is 0. Since neutral network can only process numerical inputs 
  
**3. Image pairing:**
  Since the train images are in a folder. The function `os.listdir ` is being used to read all the images file names. Then images will be paired according to their class. This allows the network to learn the key features, similarities and differences between melanoma and benign images.
  
**4. Dataset balancing / Augmentation:**
  To address the class imbalance. We set up a framework for augmenting the minority class melanoma. At this stage we are just computing how much augmentation images are needed to balance the dataset through the oversampling factor. Which can we calculate through the ratio of benign and melanoma images. Furthermore, we will also be tracking how much augmented images are being produced making sure it does not exceed the required amount of balancing
  
**5. Data Access:**
  Here we are preparing the data to be compared. By defending a class  `__getitem__`. First the first image is chosen based on the given index. Then, the network will decide whether the pair should be the same class or different class.  Last, second image will be chosen randomly from the dataset including the augmented one corresponding the given pair type 





