import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import YourDatasetClass
from model import SiameseNetwork
from torchvision import transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#Hyperparameters
batch_size = 32  #if stable change 32
learning_rate = 1e-4  
epoch = 15
margin = 1.0


# Data transforms resize my input pixel
transform = transforms.Compose([
    transforms.Resize((128, 128)), 
    transforms.ToTensor(),
])

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4) # 4 CPU

criterion = ContrastiveLoss(margin=margin)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (img1, img2, label) in enumerate(dataloader):
        img1 = img1.to(device)
        img2 = img2.to(device)
        label = label.to(device).float()

        optimizer.zero_grad() #reset gradient
        output1 = model.forward_once(img1)
        output2 = model.forward_once(img2)

        loss = criterion(output1, output2, label) 
        loss.backward() # tell weight how much need to change
        optimizer.step() # applied those change

        running_loss += loss.item()

        if i % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}] Batch [{i + 1}/{len(dataloader)}] Loss: {loss.item():.4f}')

    accuracy = correct / total
    print(f'Epoch [{epoch + 1}] Loss: {running_loss / len(dataloader):.4f} Accuracy: {accuracy:.4f}')


