import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import numpy as np

# Mount drive (if using Colab)
from google.colab import drive
drive.mount('/content/drive')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Hyperparameters
batch_size = 16
margin = 2.0
embedding_dim = 128

# Data transforms (MUST match training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create test dataset
test_dataset = SiameseDataset(
    image_dir=test_image_dir,
    labels_csv=test_labels_csv,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True if torch.cuda.is_available() else False
)

print(f'Test dataset size: {len(test_dataset)}')

# Load trained model
model = SiameseNetwork(embedding_dim=embedding_dim).to(device)
model.load_state_dict(torch.load('siamese_model.pth'))
model.eval()

criterion = ContrastiveLoss(margin=margin)

# Evaluation
total_loss = 0.0
correct = 0
total = 0
distances = []

with torch.no_grad():
    for img1, img2, label in test_loader:
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)

        # Get embeddings
        output1, output2 = model(img1, img2)

        # Calculate loss
        loss = criterion(output1, output2, label)
        total_loss += loss.item()

        # Calculate distance
        euclidean_distance = F.pairwise_distance(output1, output2)
        distances.extend(euclidean_distance.cpu().numpy())

        # Predict: if distance < threshold, same class (label=1)
        threshold = 1.0  # Tune this based on validation
        predictions = (euclidean_distance < threshold).float()

        # Calculate accuracy
        correct += (predictions == label).sum().item()
        total += label.size(0)

# Calculate metrics
test_accuracy = correct / total
avg_loss = total_loss / len(test_loader)

print(f'\n=== Test Results ===')
print(f'Test Loss: {avg_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')
print(f'Correct predictions: {correct}/{total}')
print(f'Mean distance: {np.mean(distances):.4f}')
print(f'Std distance: {np.std(distances):.4f}')
