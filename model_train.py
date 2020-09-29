import torchvision
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import numpy as np
from torchvision import transforms

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
    def __init__(self, model, num_classes=2, device='cuda:0'):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.05)
        self.criterion= nn.BCEWithLogitsLoss(reduction='mean')
        self.scheduler = ReduceLROnPlateau(
            optimizer=self.optimizer,
            factor=0.3,
            patience=1,
            verbose=True,
            mode='max',
        )
        
    def one_hot_encoder(self, num_class):
        return nn.functional.one_hot(torch.tensor(num_class),self.num_classes).numpy()
    
    def _step(self, data_loader, is_train=False):
        losses = []
        for batch in data_loader:
            img, labels = batch
            labels = torch.tensor([self.one_hot_encoder(label) for label in labels.numpy()])
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
                torch.save('classifer.pth')
            print(f'epoch: {epoch}, train_loss: {train_loss}, val_loss: {val_loss}')
            self.scheduler.step(val_loss)
        
    
def create_dataloader(root='../train_data/'):
    transformations = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = torchvision.datasets.ImageFolder(root=root, transform=transformations)
    len_train_dataset = int(len(train_data) * 0.8)
    train_set, val_set = torch.utils.data.random_split(train_data, [len_train_dataset, len(train_data)-len_train_dataset])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=8)
    return train_loader, val_loader

train_loader, val_loader = create_dataloader()
model = torchvision.models.resnext50_32x4d(pretrained=True)
model= FineTuneModel(model, num_classes=2)
model = Model(model=model)
model.train(train_loader, val_loader) 
