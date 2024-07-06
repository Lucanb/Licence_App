import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import random
import os
from sklearn.metrics import precision_score, f1_score, recall_score, confusion_matrix
import numpy as np
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

EPOCHS = 50

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
        formatId = 'ISIC_{:07d}'.format(self.ids[idx])
        newId = f'denoised_ISIC_{int(self.ids[idx]):07d}'
        datasetPath = r'C:\Users\lucan\OneDrive\Desktop\RN_TESTS\MRI_NN\data_sets\Ham10000'
        imgPath = os.path.join(datasetPath, 'images', f'{formatId}.jpg')
        maskPath = os.path.join(datasetPath, 'masks', f'{formatId}_segmentation.png')

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

model = UNet().to(device)
checkpoint = torch.load('model_checkpoint.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
trainAccuracies = checkpoint['train_accuracy']
print("train accuracies on epochs", trainAccuracies)

pixelAccuracies = []
jaccardIndexBatch = []
diceCoefficientBatch = []
precisions = []
f1_scores = []
sensitivities = []
specificities = []

with torch.no_grad():
    for inputs, labels in loadTest:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        predicted = outputs > 0.5
        labels = labels > 0.5  # Convert labels to binary

        pixelAccuracies.append(pixel_accuracy(predicted, labels))
        jaccardIndexBatch.append(jaccard_index(predicted, labels))
        diceCoefficientBatch.append(dice_coefficient(predicted, labels))
        precisions.append(precision(predicted, labels))
        f1_scores.append(f1(predicted, labels))
        sensitivities.append(recall(predicted, labels))
        specificities.append(specificity(predicted, labels))

metrics = {
    'Pixel Accuracy': pixelAccuracies,
    'Jaccard Index': jaccardIndexBatch,
    'Dice Coefficient': diceCoefficientBatch,
    'Precision': precisions,
    'F1 Score': f1_scores,
    'Sensitivity': sensitivities,
    'Specificity': specificities
}

metricsHash = {}
for nameMetric, values in metrics.items():
    meanVal = np.mean(values)
    stdVal = np.std(values)
    metricsHash[nameMetric] = (meanVal, stdVal)
    print(f"{nameMetric}: Mean = {meanVal:.4f}, Std Dev = {stdVal:.4f}")

with open('test.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Batch', 'Pixel Accuracy', 'Jaccard Index', 'Dice Coefficient', 'Precision', 'F1 Score', 'Sensitivity', 'Specificity'])
    for i in range(len(pixelAccuracies)):
        writer.writerow([i+1, pixelAccuracies[i], jaccardIndexBatch[i], diceCoefficientBatch[i], precisions[i], f1_scores[i], sensitivities[i], specificities[i]])

    writer.writerow([])
    writer.writerow(['Metric', 'Mean', 'Standard Deviation'])
    for nameMetric, (meanVal, stdVal) in metricsHash.items():
        writer.writerow([nameMetric, meanVal, stdVal])

testAccuracies = []
for epoch in range(EPOCHS):
    print(f'Epoch: {epoch + 1}')
    saveDir = f'checkpoints/model_checkpoint_epoch{epoch + 1}.pth'
    if os.path.exists(saveDir):
        train_results_checkpoint = torch.load(saveDir)
        model.load_state_dict(train_results_checkpoint['model_state_dict'])
        model.eval()
        total_correct_test = 0
        total_test_samples = 0

        with torch.no_grad():
            for inputs, labels in loadTest:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                predicted = outputs > 0.5
                labels = labels > 0.5
                
                total_correct_test += (predicted == labels).sum().item()
                total_test_samples += labels.numel()

        test_accuracy = total_correct_test / total_test_samples
        testAccuracies.append(test_accuracy)

plt.figure(figsize=(10, 5))
print('Test accuracies on epochs:', testAccuracies)
plt.plot(trainAccuracies, label='Train Accuracy', color='blue')
plt.plot(testAccuracies, label='Test Accuracy', color='red')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train and Test Accuracy per Epoch')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('train_and_test_accuracies2_denoise.png')
