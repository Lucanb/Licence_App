import os
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, f1_score, recall_score, confusion_matrix
import numpy as np
import csv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

EPOCHS = 25

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

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

class Ham10000Dataset(Dataset):
    def __init__(self, ids, datasetPath, transform=None):
        self.ids = ids
        self.datasetPath = datasetPath
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        formatted_id = 'ISIC_{:07d}'.format(self.ids[idx])
        image_path = os.path.join(self.datasetPath, 'images', f'{formatted_id}.jpg')
        mask_path = os.path.join(self.datasetPath, 'masks', f'{formatted_id}_segmentation.png')

        try:
            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
            if self.transform:
                image = self.transform(image)
                mask = self.transform(mask)
            return image, mask
        except FileNotFoundError as e:
            print(f"File not found: {e}")
            return None, None

TRAIN_SIZE = 7000
TEST_SIZE = 3000
DATASETPATH = r'C:\Users\lucan\OneDrive\Desktop\RN_TESTS\MRI_NN\data_sets\Ham10000'
allID = random.sample(range(24306, 34321), TRAIN_SIZE + TEST_SIZE)
trainID = allID[:TRAIN_SIZE]
testID = allID[TRAIN_SIZE:]

trainData = Ham10000Dataset(ids=trainID, datasetPath=DATASETPATH, transform=transform)
testData = Ham10000Dataset(ids=testID, datasetPath=DATASETPATH, transform=transform)

loadTrain = DataLoader(trainData, batch_size=64, shuffle=True)
loadTest = DataLoader(testData, batch_size=64, shuffle=False)

class FullyCNET(nn.Module):
    def __init__(self, num_classes):
        super(FullyCNET, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x64
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = FullyCNET(num_classes=1).to(device)

saveDir = 'model_checkpoint.pth'
if not os.path.exists(saveDir):
    raise FileNotFoundError(f"No checkpoint file found at '{saveDir}'. Please ensure the path is correct and the file exists.")

checkpoint = torch.load(saveDir, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
trainAccuracies = checkpoint['train_accuracy']
print("Train accuracies on epochs:", trainAccuracies)

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
        labels = labels > 0.5

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

with open('evaluation_metrics_fcn.csv', mode='w', newline='') as file:
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
    saveDir = f'checkpoints/model_checkpoint_epoch_{epoch + 1}.pth'
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
plt.savefig('train_and_test_accuracies_fcn.png')
