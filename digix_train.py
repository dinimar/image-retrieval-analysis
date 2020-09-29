from os import listdir
from os.path import isfile, join

import torch
from torch import nn
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from FixRes.imnet_extract.resnext_wsl import resnext101_32x48d_wsl

import csv
import datetime
from PIL import Image
from PIL import ImageFile
import albumentations as A

ImageFile.LOAD_TRUNCATED_IMAGES = True

class DIGIXNet(nn.Module):
    def __init__(self):
        super(DIGIXNet, self).__init__()
        # input - decriptor of length 2048
        # ouput - 3097 out classes
        self.fc = nn.Linear(2048, 3097)

    def forward(self, x):
        x = self.fc(x)
        
        return x


TRAIN = '../train_data'

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using cuda')


def image_loader(filename):
    input_image = Image.open(filename).convert('RGB')

    aug = A.Compose([
        A.RandomCrop(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ])
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    aug_image = aug(image=input_image)['image']            # add augmentation
    input_tensor = preprocess(aug_image)    # convert to tensor

    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    input_batch = input_batch.to(device)

    return input_batch


model=resnext101_32x48d_wsl(progress=True) # example with the ResNeXt-101 32x48d 
pretrained_dict=torch.load('../ResNext101_32x48d_v2.pth',map_location='cuda:0')['model']

model_dict = model.state_dict()
for k in model_dict.keys():
    if(('module.'+k) in pretrained_dict.keys()):
        model_dict[k]=pretrained_dict.get(('module.'+k))
model.load_state_dict(model_dict)
model.eval()
model.to(device)

# freeze layers to prevent retraining
for parameter in model.parameters():
    parameter.requires_grad = False


# extract feauture vectors from images
print(datetime.datetime.now().time(), 'Creating training dataset')
train_data = []
with open(TRAIN+'/label.txt', newline='') as csvfile:
    rows = csv.reader(csvfile, delimiter=',', quotechar='|')
    train_rows = list(rows)

    for row in train_rows:
        try:
            img_path = TRAIN+'/'+row[0]
            input_batch = image_loader(img_path)

            # descriptor tensor 2048
            desc_tensor = model(input_batch)
            # output class tensor with 3097 classes
            labels = torch.zeros([1, 3097], dtype=torch.float32, device=device)
            labels[0][int(row[1])] = 1

            train_data.append((desc_tensor, labels))
        except OSError as e:
            print(e)
            print('Bad image', row[0])
print(datetime.datetime.now().time(), 'Training dataset created')

# train network
net = DIGIXNet()
net.to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=0.05)
criterion= nn.BCEWithLogitsLoss(reduction='mean')

print(datetime.datetime.now().time(), 'Start training')
for epoch in range(1):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_data, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print(datetime.datetime.now().time(), 'Finished Training')

# save weights
PATH = './digix_net.pth'
torch.save(net.state_dict(), PATH)
print(datetime.datetime.now().time(), 'Saved weights')