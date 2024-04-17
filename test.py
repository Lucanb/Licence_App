import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import random

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

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.decoder(x1)
        return x2

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
        image = Image.open('dataset/images/ISIC_00{}.jpg'.format(self.ids[idx]))
        mask = Image.open('dataset/masks/img/ISIC_00{}_segmentation.png'.format(self.ids[idx]))
        if self.transform:
            image = self.transform(image)
            mask = transforms.Resize((128, 128))(mask)
            mask = transforms.ToTensor()(mask)
        return image, mask

TEST_SIZE = 3000
BATCH_SIZE = 64

test_dataset_ids = random.sample(range(24306, 34321), TEST_SIZE)
test_dataset = CustomDataset(test_dataset_ids, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

checkpoint = torch.load('model_checkpoint.pth')

model = UNet()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
train_accuracies = checkpoint['train_accuracy'] # aici mi-am salvat acuratetea pentru ficare epoca in modelul final
print("train accuracies on epochs",train_accuracies)
pixel_accuracies = []
jaccard_indexes = []
dice_coefficients = []

test_accuracies = []
total_correct_test = 0
total_test_samples = 0

# Aici definim lista 'batches' după ce am calculat rezultatele pentru fiecare batch
batches = range(1, len(pixel_accuracies) + 1)

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        predicted = outputs > 0.5
        total_correct_test += (predicted == labels).sum().item()
        total_test_samples += labels.numel()
        pixel_accuracies.append(pixel_accuracy(predicted, labels))
        jaccard_indexes.append(jaccard_index(predicted, labels))
        dice_coefficients.append(dice_coefficient(predicted, labels))

        # Calculăm și stocăm rezultatele pentru fiecare batch în 'batches'
        batches = range(1, len(pixel_accuracies) + 1)

# După ce am terminat de calculat rezultatele pentru fiecare batch, putem trasa graficul
plt.plot(batches, pixel_accuracies, label='Pixel Accuracy')
plt.plot(batches, jaccard_indexes, label='Jaccard Index')
plt.plot(batches, dice_coefficients, label='Dice Coefficient')
plt.xlabel('Batch')
plt.ylabel('Value')
plt.title('Evaluation Metrics per Batch')
plt.legend()
plt.grid(True)
plt.savefig('evaluation_metrics.png')


#pe asta fac analiza de overfitt
for epoch in range(EPOCHS):
    print(epoch)
    train_results_checkpoint = torch.load('checkpoints/model_checkpoint_epoch{}.pth'.format(epoch))
    model_overfitt = UNet()
    model_overfitt.load_state_dict(checkpoint['model_state_dict'])
    model_overfitt.eval()
    total_correct_test = 0
    total_test_samples = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model_overfitt(inputs)
            predicted = outputs > 0.5
            total_correct_test += (predicted == labels).sum().item()
            total_test_samples += labels.numel()

    test_accuracy = total_correct_test / total_test_samples
    test_accuracies.append(test_accuracy)
    


plt.figure(figsize=(10, 5))
print('test accuracies on epochs',test_accuracies)
plt.plot(train_accuracies, label='Train Accuracy', color='blue')
plt.plot(test_accuracies, label='Test Accuracy', color='red')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train and Test Accuracy per Epoch')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('train_and_test_accuracies.png')