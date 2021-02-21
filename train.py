import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
import os
from datasetload import Cardataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)

learning_rate = 1e-3
batch_size = 128
num_epochs = 10

csv_file = pd.DataFrame()
count = 0
for root, dirs, files in os.walk('imgs_zip\imgs'):
    for file in files:
        csv_file.loc[count, 'file'] = file
        csv_file.loc[count, 'class'] = root.split('\\')[2]
        count += 1
num_classes = csv_file['class'].nunique()
le = preprocessing.LabelEncoder()
le.fit(csv_file['class'])
csv_file['class'] = le.transform(csv_file['class'])

model = torchvision.models.mobilenet_v2(pretrained=True)

for param in model.parameters():
    param.requires_grad = False
# print(model)
num_features = model.classifier[1].in_features
features = list(model.classifier.children())[:-1] # Remove last layer
features.extend([nn.Linear(num_features, num_classes)]) # Add our layer with 4 outputs
model.classifier = nn.Sequential(*features)
# print(model)
model.to(device)

dataset = Cardataset(csv_file=csv_file,root_dir='imgs_zip\images',transform=transforms.Compose([
                                                                                        transforms.ToPILImage(),
                                                                                        transforms.Resize((224,224)),
                                                                                        transforms.ToTensor()
                                                                                    ]))
train_set, test_set = torch.utils.data.random_split(dataset, [3678, 919])

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

train_losses = []
valid_losses = []

for epoch in range(num_epochs):
    train_loss = 0.0
    valid_loss = 0.0

    model.train()
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()
        train_loss += loss.item() * data.size(0)

    model.eval()
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)

        output = model(data)

        loss = criterion(output, target)

        # update-average-validation-loss
        valid_loss += loss.item() * data.size(0)

    train_loss = train_loss / len(train_loader.sampler)
    valid_loss = valid_loss / len(test_loader.sampler)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch+1, train_loss, valid_loss))
print("Saving the model")
torch.save({'state_dict': model.state_dict()}, 'model_checkpoint.pth')

model.eval()  # it-disables-dropout
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model: {} %'.format(100 * correct / total))


plt.plot(train_losses, label='Training loss')
plt.plot(valid_losses, label='Validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(frameon=False)
plt.savefig('train_test_loss.jpg')

