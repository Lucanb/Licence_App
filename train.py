import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
import numpy as np
from PIL import Image
import os
from imageAjust import CustomPixelManipulation
from watershed import  watershed_segmentation

TRAIN_SIZE = 7000   
TEST_SIZE = 3000
BATCH_SIZE = 64
EPOCHS = 50
BATCHES_PER_EPOCH = 10

# Now integrate this into the existing transformation pipeline
transform_augmented = transforms.Compose([
    transforms.Resize((128, 128)),
    CustomPixelManipulation(50),  # Apply custom pixel manipulation
    transforms.ToTensor(),  # Convert the modified image to a tensor
])

# transform_augmented = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
#     transforms.RandomRotation(10),       # Random rotation of ±10 degrees
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random color jitter
#     transforms.ToTensor(),
# ])


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

class CustomDataset(Dataset):
    def __init__(self, ids, transform=None):
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image = Image.open('dataset/img1/ISIC_00{}.jpg'.format(self.ids[idx]))
        mask = Image.open('dataset/mask1/ISIC_00{}_segmentation.png'.format(self.ids[idx]))
        if self.transform:
            image = self.transform(image)
            mask = transforms.Resize((128, 128))(mask)
            mask = transforms.ToTensor()(mask)
        return image, mask

dataset_ids = random.sample(range(24306, 34321), TRAIN_SIZE + TEST_SIZE)
train_dataset_ids = dataset_ids[:TRAIN_SIZE]
test_dataset_ids = dataset_ids[TRAIN_SIZE:]

train_dataset = CustomDataset(train_dataset_ids, transform=transform_augmented)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = CustomDataset(test_dataset_ids, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # self.encoder = nn.Sequential(
        #     nn.Conv2d(1, 64, kernel_size=3, padding=1),  # Modificat de la 3 la 1 canal
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        # )
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 1, kernel_size=3, padding=1),
        #     nn.Sigmoid()
        # )
    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.decoder(x1)
        return x2
    
model = UNet()

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())
train_accuracies = []

for epoch in range(EPOCHS):
    running_loss = 0.0
    total_correct_train = 0
    total_train_samples = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        predicted = outputs > 0.5
        total_correct_train += (predicted == labels).sum().item()
        total_train_samples += labels.numel()
        
        if i % BATCHES_PER_EPOCH == BATCHES_PER_EPOCH - 1:
            print("[{}, {}] loss: {:.6f}".format(epoch + 1, i + 1, running_loss / BATCHES_PER_EPOCH))
            running_loss = 0.0

    train_accuracy = total_correct_train / total_train_samples
    train_accuracies.append(train_accuracy)
    checkpoint_file = os.path.join('checkpoints', 'model_checkpoint_epoch{}.pth'.format(epoch))
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'train_accuracy': train_accuracy
    }, checkpoint_file)

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': loss,
    'train_accuracy': train_accuracies
}, 'model_checkpoint.pth')

checkpoint = torch.load('model_checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']


print("Finished training")
