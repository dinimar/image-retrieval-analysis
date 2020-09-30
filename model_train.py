import argparse
import torchvision
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import numpy as np
from torchvision import transforms
from dataset import ClassifierDataset

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from dataset import create_dataloader

class FineTuneModel(nn.Module):
    def __init__(self, original_model, num_classes):
        super(FineTuneModel, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.classifier = nn.Linear(2048, num_classes)
        self.modelName = 'Resnet50'
        for p in self.features.parameters():
            p.requires_grad = False


    def forward(self, x):
        f = self.features(x)        
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y
    
class Model:
    def __init__(self, model, device='cuda:0'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.05)
        self.criterion= nn.BCEWithLogitsLoss(reduction='mean')
        self.scheduler = ReduceLROnPlateau(
            optimizer=self.optimizer,
            factor=0.3,
            patience=1,
            verbose=True,
            mode='max',
        )
    
    def _step(self, data_loader, is_train=False):
        losses = []
        for batch in data_loader:
            img, labels = batch.values()
            img, labels= (
                img.to(self.device),
                labels.to(self.device),
            )
            model_out = self.model.forward(img)
            loss = self.criterion(model_out, labels.type_as(model_out))
            losses.append(loss.item())
            if is_train:
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
        return np.mean(losses)
    
    def train(self, train_dataloader, val_dataloader, save_folder='./checkpoints/', num_epochs=240):
        best_loss = 100
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = self._step(
                train_dataloader,
                is_train=True,
            )
            with torch.no_grad():
                val_loss = self._step(
                    val_dataloader,
                )
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model, 'classifer.pth')
            print(f'epoch: {epoch}, train_loss: {train_loss}, val_loss: {val_loss}')
            self.scheduler.step(val_loss)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train loop for classification net.')
    parser.add_argument('num_classes', type=int, help='a number of classes in dataset')
    parser.add_argument('csv_file', type=str, help='path to csv file that contains dataset info in format:\n1.jpg,0\n2.jpg,1')
    parser.add_argument('root_dir', type=str, help='path to root dir with dataset')
    args = parser.parse_args()
    
    num_classes = args.num_classes
    csv_file = args.csv_file
    root_dir = args.root_dir

    train_loader, val_loader = create_dataloader(
        csv_file=csv_file,
        root_dir=root_dir,
        num_classes=num_classes
    )
    model = torchvision.models.resnext50_32x4d(pretrained=True)
    model= FineTuneModel(model, num_classes=num_classes)
    model = Model(model=model)
    model.train(train_loader, val_loader) 
