import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

BATCH_SIZE = 64
EPOCHS = 25

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
        imgPath = os.path.join(self.datasetPath, 'images', f'{formatted_id}.jpg')
        maskPath = os.path.join(self.datasetPath, 'masks', f'{formatted_id}_segmentation.png')

        try:
            image = Image.open(imgPath).convert("RGB")
            mask = Image.open(maskPath).convert("L")
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

loadTrain = DataLoader(trainData, batch_size=BATCH_SIZE, shuffle=True)
loadTest = DataLoader(testData, batch_size=BATCH_SIZE, shuffle=False)

class FullyCNET(nn.Module):
    def __init__(self, num_classes):
        super(FullyCNET, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
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

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

saveDir = './checkpoints'

if not os.path.exists(saveDir):
    os.makedirs(saveDir)

trainAccuracies = []

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    total_correct_train = 0
    total_train_samples = 0
    for images, masks in loadTrain:
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
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

    print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')

    checkpoint_path = os.path.join(saveDir, f'model_checkpoint_epoch_{epoch+1}.pth')
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_train_loss,
        'train_accuracy': train_accuracy,
    }, checkpoint_path)

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': EPOCHS,
    'loss': avg_train_loss,
    'train_accuracy': trainAccuracies,
}, 'model_checkpoint.pth')

checkpoint = torch.load('model_checkpoint.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

print("Finished training")
