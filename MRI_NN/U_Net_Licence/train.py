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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

BATCH_SIZE = 64
EPOCHS = 50
BATCHES_PER_EPOCH = 10

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

class HamDataset(Dataset):
    def __init__(self, ids, transform=None):
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        formatted_id = 'ISIC_{:07d}'.format(self.ids[idx])
        datasetPath = r'C:\Users\lucan\OneDrive\Desktop\RN_TESTS\MRI_NN\data_sets\Ham10000'
        imgPath = os.path.join(datasetPath, 'images', f'{formatted_id}.jpg')
        maskPath = os.path.join(datasetPath, 'masks', f'{formatted_id}_segmentation.png')

        try:
            image = Image.open(imgPath)
            mask = Image.open(maskPath)
            if self.transform:
                image = self.transform(image)
                mask = self.transform(mask)
            return image, mask
        except FileNotFoundError as e:
            print(f"File not found: {e}")
            return None, None

TRAIN_SIZE = 7000
TEST_SIZE = 3000
allID = random.sample(range(24306, 34321), TRAIN_SIZE + TEST_SIZE)
trainID = allID[:TRAIN_SIZE]
testID = allID[TRAIN_SIZE:]

trainData = HamDataset(ids=trainID, transform=transform)
testData = HamDataset(ids=testID, transform=transform)

loadTrain = DataLoader(trainData, batch_size=BATCH_SIZE, shuffle=True)
loadTest = DataLoader(testData, batch_size=BATCH_SIZE, shuffle=False)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc_conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(64)
        self.enc_relu1 = nn.ReLU(inplace=True)
        self.enc_pool1 = nn.MaxPool2d(2)
        
        self.enc_conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(128)
        self.enc_relu2 = nn.ReLU(inplace=True)
        self.enc_pool2 = nn.MaxPool2d(2)
        
        self.enc_conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.enc_bn3 = nn.BatchNorm2d(256)
        self.enc_relu3 = nn.ReLU(inplace=True)
        self.enc_pool3 = nn.MaxPool2d(2)
        
        self.bottleneck_conv = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bottleneck_bn = nn.BatchNorm2d(512)
        self.bottleneck_relu = nn.ReLU(inplace=True)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec_conv3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.dec_bn3 = nn.BatchNorm2d(256)
        self.dec_relu3 = nn.ReLU(inplace=True)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.dec_bn2 = nn.BatchNorm2d(128)
        self.dec_relu2 = nn.ReLU(inplace=True)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dec_bn1 = nn.BatchNorm2d(64)
        self.dec_relu1 = nn.ReLU(inplace=True)
        
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.final_activation = nn.Sigmoid()

    def forward(self, x):
        enc1 = self.enc_relu1(self.enc_bn1(self.enc_conv1(x)))
        enc1_pool = self.enc_pool1(enc1)
        
        enc2 = self.enc_relu2(self.enc_bn2(self.enc_conv2(enc1_pool)))
        enc2_pool = self.enc_pool2(enc2)
        
        enc3 = self.enc_relu3(self.enc_bn3(self.enc_conv3(enc2_pool)))
        enc3_pool = self.enc_pool3(enc3)
        
        bottleneck = self.bottleneck_relu(self.bottleneck_bn(self.bottleneck_conv(enc3_pool)))
        
        dec3_up = self.upconv3(bottleneck)
        dec3 = self.dec_relu3(self.dec_bn3(self.dec_conv3(torch.cat([dec3_up, enc3], dim=1))))
        
        dec2_up = self.upconv2(dec3)
        dec2 = self.dec_relu2(self.dec_bn2(self.dec_conv2(torch.cat([dec2_up, enc2], dim=1))))
        
        dec1_up = self.upconv1(dec2)
        dec1 = self.dec_relu1(self.dec_bn1(self.dec_conv1(torch.cat([dec1_up, enc1], dim=1))))
        
        out = self.final_activation(self.final_conv(dec1))
        
        return out

model = UNet().to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())
trainAccuracies = []

for epoch in range(EPOCHS):
    running_loss = 0.0
    total_correct_train = 0
    total_train_samples = 0
    for i, data in enumerate(loadTrain, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
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
    trainAccuracies.append(train_accuracy)
    saveDir = os.path.join('checkpoints', 'model_checkpoint_epoch{}.pth'.format(epoch))
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'train_accuracy': train_accuracy
    }, saveDir)

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': loss,
    'train_accuracy': trainAccuracies
}, 'model_checkpoint.pth')

checkpoint = torch.load('model_checkpoint.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

print("Finished training")
