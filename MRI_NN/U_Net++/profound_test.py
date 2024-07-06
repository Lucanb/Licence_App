import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
from sklearn.metrics import precision_score, f1_score, recall_score, confusion_matrix
import numpy as np
import csv

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
        x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=True)
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

        self.ds_outc4 = nn.Conv2d(256, 1, kernel_size=1)
        self.ds_outc3 = nn.Conv2d(128, 1, kernel_size=1)
        self.ds_outc2 = nn.Conv2d(64, 1, kernel_size=1)
        self.ds_outc31 = nn.Conv2d(128, 1, kernel_size=1)
        self.ds_outc21 = nn.Conv2d(64, 1, kernel_size=1)

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

        ds_logits4 = self.ds_outc4(x4_up)
        ds_logits3 = self.ds_outc3(x3_up)
        ds_logits2 = self.ds_outc2(x2_up)
        ds_logits31 = self.ds_outc31(x3_1)
        ds_logits21 = self.ds_outc21(x2_1)

        return logits, ds_logits4, ds_logits3, ds_logits2, ds_logits31, ds_logits21

image_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mask_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

class HamDataset(Dataset):
    def __init__(self, ids, image_transform=None, mask_transform=None, datasetPath=r'C:\Users\lucan\OneDrive\Desktop\RN_TESTS\MRI_NN\data_sets\Ham10000'):
        self.ids = ids
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.datasetPath = datasetPath

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        formatted_id = f'ISIC_{int(self.ids[idx]):07d}'
        imgPath = os.path.join(self.datasetPath, 'images', f'{formatted_id}.jpg')
        maskPath = os.path.join(self.datasetPath, 'masks', f'{formatted_id}_segmentation.png')

        image = Image.open(imgPath).convert('RGB')
        mask = Image.open(maskPath).convert('L')
        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNetPlusPlus().to(device)
model_path = 'unetplusplus_supervised_final.pth'
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

testID = [i for i in range(24306, 27306)]
testData = HamDataset(ids=testID, image_transform=image_transform, mask_transform=mask_transform)
loadTest = DataLoader(testData, batch_size=16, shuffle=False)

final_checkpoint = torch.load('unetplusplus_supervised_final.pth', map_location=device)
model.load_state_dict(final_checkpoint['model_state_dict'])
trainAccuracies = final_checkpoint['train_accuracy']

def pixel_accuracy(predicted, target):
    correct_pixels = (predicted == target).sum().item()
    total_pixels = target.numel()
    return correct_pixels / total_pixels

def jaccard_index(predicted, target):
    intersection = torch.logical_and(predicted, target).sum().item()
    union = torch.logical_or(predicted, target).sum().item()
    return intersection / union

def dice_coefficient(predicted, target):
    intersection = torch.logical_and(predicted, target).sum().item()
    dice = (2 * intersection) / (predicted.sum().item() + target.sum().item())
    return dice

def precision(predicted, target):
    pred_flat = predicted.view(-1).cpu().numpy()
    target_flat = target.view(-1).cpu().numpy()
    return precision_score(target_flat, pred_flat, average='binary')

def f1(predicted, target):
    pred_flat = predicted.view(-1).cpu().numpy()
    target_flat = target.view(-1).cpu().numpy()
    return f1_score(target_flat, pred_flat, average='binary')

def recall(predicted, target):
    pred_flat = predicted.view(-1).cpu().numpy()
    target_flat = target.view(-1).cpu().numpy()
    return recall_score(target_flat, pred_flat, average='binary')

def specificity(predicted, target):
    cm = confusion_matrix(target.view(-1).cpu().numpy(), predicted.view(-1).cpu().numpy())
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp)

metrics = {
    'Pixel Accuracy': [],
    'Jaccard Index': [],
    'Dice Coefficient': [],
    'Precision': [],
    'F1 Score': [],
    'Sensitivity': [],
    'Specificity': []
}

with torch.no_grad():
    for inputs, labels in loadTest:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs, _, _, _, _, _ = model(inputs)
        predicted = outputs.sigmoid() > 0.5
        predicted = predicted.float()  # Ensure predicted values are binary (0 or 1)
        labels = labels.float() > 0.5  # Ensure labels are binary (0 or 1)

        metrics['Pixel Accuracy'].append(pixel_accuracy(predicted, labels))
        metrics['Jaccard Index'].append(jaccard_index(predicted, labels))
        metrics['Dice Coefficient'].append(dice_coefficient(predicted, labels))
        metrics['Precision'].append(precision(predicted, labels))
        metrics['F1 Score'].append(f1(predicted, labels))
        metrics['Sensitivity'].append(recall(predicted, labels))
        metrics['Specificity'].append(specificity(predicted, labels))

metricsHash = {name: (np.mean(values), np.std(values)) for name, values in metrics.items()}
for name, (mean_val, std_val) in metricsHash.items():
    print(f"{name}: Mean = {mean_val:.4f}, Std Dev = {std_val:.4f}")

with open('evaluation_metrics_supervised.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Batch', 'Pixel Accuracy', 'Jaccard Index', 'Dice Coefficient', 'Precision', 'F1 Score', 'Sensitivity', 'Specificity'])
    for i in range(len(metrics['Pixel Accuracy'])):
        writer.writerow([
            i + 1,
            metrics['Pixel Accuracy'][i],
            metrics['Jaccard Index'][i],
            metrics['Dice Coefficient'][i],
            metrics['Precision'][i],
            metrics['F1 Score'][i],
            metrics['Sensitivity'][i],
            metrics['Specificity'][i]
        ])
    writer.writerow([])
    writer.writerow(['Metric', 'Mean', 'Standard Deviation'])
    for name, (mean_val, std_val) in metricsHash.items():
        writer.writerow([name, mean_val, std_val])

testAccuracies = []
EPOCHS = 50
for epoch in range(EPOCHS):
    checkpoint_path = f'checkpoints_supervised/model_checkpoint_epoch{epoch+1}.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        total_correct_test = 0
        total_test_samples = 0
        with torch.no_grad():
            for inputs, labels in loadTest:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, _, _, _, _, _ = model(inputs)
                predicted = outputs.sigmoid() > 0.5
                predicted = predicted.float()
                labels = labels.float() > 0.5
                total_correct_test += (predicted == labels).sum().item()
                total_test_samples += labels.numel()

        test_accuracy = total_correct_test / total_test_samples
        testAccuracies.append(test_accuracy)
        print(f'Epoch {epoch+1}: Test Accuracy = {test_accuracy:.4f}')

if trainAccuracies and testAccuracies:
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS + 1), trainAccuracies, label='Train Accuracy', color='blue')
    plt.plot(range(1, EPOCHS + 1), testAccuracies, label='Test Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train and Test Accuracy per Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('train_and_testAccuracies_supervised.png')
    plt.show()
else:
    print("No data available for plotting.")
