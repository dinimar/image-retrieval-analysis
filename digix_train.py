from os import listdir
from os.path import isfile, join

import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms
from FixRes.imnet_extract.resnext_wsl import resnext101_32x48d_wsl

from PIL import Image

TRAIN = 'cars_train/cars_train'

def image_loader(filename):
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')

    return input_batch


model=resnext101_32x48d_wsl(progress=True) # example with the ResNeXt-101 32x48d 

pretrained_dict=torch.load('ResNext101_32x48d_v2.pth',map_location='cuda:0')['model']

model_dict = model.state_dict()
for k in model_dict.keys():
    if(('module.'+k) in pretrained_dict.keys()):
        model_dict[k]=pretrained_dict.get(('module.'+k))
model.load_state_dict(model_dict)
model.eval()

# extract feauture vectors from images
images = [join(TRAIN, f) for f in listdir(TRAIN) if isfile(join(TRAIN, f))]

if torch.cuda.is_available():
    model.to('cuda')

for img in images:
    input_batch = image_loader(img)

    with torch.no_grad():
        output = model(input_batch)

    print(output[0])
    print(output[0].size())

    break
