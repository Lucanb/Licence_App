import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

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

loadTrain = DataLoader(trainData, batch_size=64, shuffle=True)
loadTest = DataLoader(testData, batch_size=64, shuffle=False)

class DeepLabNN(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabNN, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x64

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 8x8
        )

        self.aspp = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=6, dilation=6),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=12, dilation=12),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=18, dilation=18),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.backbone(x)
        x = self.aspp(x)
        x = self.decoder(x)
        x = self.upsample(x)
        return x

model = DeepLabNN(num_classes=1).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 50
saveDir = './checkpoints'
if not os.path.exists(saveDir):
    os.makedirs(saveDir)

trainAccuracies = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    total_correct_train = 0
    total_train_samples = 0
    for images, masks in loadTrain:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        masks = nn.functional.interpolate(masks, size=outputs.shape[2:], mode='bilinear', align_corners=False)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        predicted = outputs > 0.5
        total_correct_train += (predicted == masks).sum().item()
        total_train_samples += masks.numel()

    avg_train_loss = train_loss / len(loadTrain)
    train_accuracy = total_correct_train / total_train_samples
    trainAccuracies.append(train_accuracy)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')

    checkpoint_path = os.path.join(saveDir, f'model_checkpoint_epoch_{epoch+1}.pth')
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_train_loss,
        'train_accuracy': train_accuracy,
    }, checkpoint_path)

torch.save({
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': avg_train_loss,
    'train_accuracy': trainAccuracies,
}, 'model_checkpoint.pth')

print("Training complete")
