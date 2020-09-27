from os import listdir
from os.path import isfile, join

import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms
from FixRes.imnet_extract.resnext_wsl import resnext101_32x48d_wsl

from PIL import Image

TRAIN = 'cars_train/cars_train'

def image_loader(loader, image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = loader(image).float()
    #image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image  #assumes that you're using GPU

model=resnext101_32x48d_wsl(progress=True) # example with the ResNeXt-101 32x48d 

pretrained_dict=torch.load('ResNext101_32x48d_v2.pth',map_location='cuda:0')['model']

model_dict = model.state_dict()
for k in model_dict.keys():
    if(('module.'+k) in pretrained_dict.keys()):
        model_dict[k]=pretrained_dict.get(('module.'+k))
model.load_state_dict(model_dict)

feat_ext_model = nn.Sequential(*list(model.modules())[:-1])

# extract feauture vectors from images
loader = transforms.Compose([ transforms.ToTensor()])
# train_set =datasets.ImageFolder('train')


onlyfiles = [f for f in listdir(TRAIN) if isfile(join(TRAIN, f))]
print(onlyfiles)

