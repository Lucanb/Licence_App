import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

class HamDataset(Dataset):
    def __init__(self, ids, image_transform=None, mask_transform=None):
        self.ids = ids
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        formatted_id = f'ISIC_{int(self.ids[idx]):07d}'
        datasetPath = r'C:\Users\lucan\OneDrive\Desktop\RN_TESTS\MRI_NN\data_sets\Ham10000'
        imgPath = os.path.join(datasetPath, 'images', f'{formatted_id}.jpg')
        maskPath = os.path.join(datasetPath, 'masks', f'{formatted_id}_segmentation.png')

        image = Image.open(imgPath).convert('RGB')
        mask = Image.open(maskPath).convert('L')

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels * 2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = F.interpolate(x1, size=x2.shape[2:])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class NestedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NestedConvBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        return self.conv(x)

class UNetPlusPlus(nn.Module):
    def __init__(self):
        super(UNetPlusPlus, self).__init__()
        self.inc = ConvBlock(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        self.up1_0 = Up(512, 256)
        self.up2_0 = Up(256, 128)
        self.up3_0 = Up(128, 64)

        self.up2_1 = NestedConvBlock(512, 128)
        self.up3_1 = NestedConvBlock(256, 64)
        self.up3_2 = NestedConvBlock(128, 64)

        self.outc = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x4_up = self.up1_0(x4, x3)
        x3_up = self.up2_0(x4_up, x2)
        x2_up = self.up3_0(x3_up, x1)

        x3_1 = self.up2_1(torch.cat([x4_up, x3], dim=1))
        x2_1 = self.up3_1(torch.cat([x3_up, x2], dim=1))

        x3_2 = self.up3_2(torch.cat([x2_up, x1], dim=1))

        logits = self.outc(x3_2)
        return logits

image_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
mask_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

TRAIN_SIZE = 7000
TEST_SIZE = 3000
allID = random.sample(range(24306, 34321), TRAIN_SIZE + TEST_SIZE)
trainID = allID[:TRAIN_SIZE]
testID = allID[TRAIN_SIZE:]
trainData = HamDataset(ids=trainID, image_transform=image_transform, mask_transform=mask_transform)
testData = HamDataset(ids=testID, image_transform=image_transform, mask_transform=mask_transform)
BATCH_SIZE = 16
loadTrain = DataLoader(trainData, batch_size=BATCH_SIZE, shuffle=True)
loadTest = DataLoader(testData, batch_size=BATCH_SIZE, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNetPlusPlus().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')

NUM_EPOCHS = 50
trainAccuracies = []
for epoch in range(NUM_EPOCHS):
    model.train()
    total_correct_train = 0
    total_train_samples = 0
    for images, masks in loadTrain:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        predicted = outputs.sigmoid() > 0.5
        correct = (predicted == masks).float().sum()
        total_correct_train += correct
        total_train_samples += masks.numel()

    train_accuracy = total_correct_train / total_train_samples
    trainAccuracies.append(train_accuracy.item())

    saveDir = os.path.join('checkpoints', f'model_checkpoint_epoch{epoch+1}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss.item(),
        'train_accuracy': train_accuracy.item()
    }, saveDir)
    print(f'Epoch {epoch+1}: Loss = {loss.item()}, Train Accuracy = {train_accuracy.item()}')

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': loss,
    'train_accuracy': trainAccuracies
}, 'unetplusplus_final.pth')

print("Modelul final a fost salvat.")
