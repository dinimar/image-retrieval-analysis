import argparse
import torchvision
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import average_precision_score, accuracy_score
import torch
from PIL import Image
import numpy as np

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
    
    
class ModelInterface:
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
        
    def _one_hot2labels(self, one_hot):
        return np.argmax(one_hot, axis=1)
        
    def _compute_accuracy(self, pred, target):
        pred, target = pred.cpu().detach().numpy(), target.cpu().detach().numpy()
        pred, target = self._one_hot2labels(pred), self._one_hot2labels(target)
        return accuracy_score(target, pred)
        
    def _compute_map(self, pred, target):
        pred, target = pred.cpu().detach().numpy(), target.cpu().detach().numpy()
        return np.mean([average_precision_score(target[:, i], pred[:, i]) for i in range(len(pred[0]))])
    
    def _step(self, data_loader, is_train=False):
        losses = []
        mAP = []
        acc = []
        for batch in data_loader:
            img, labels = batch.values()
            img, labels= (
                img.to(self.device),
                labels.to(self.device),
            )
            model_out = self.model.forward(img)
            self._compute_accuracy(model_out, labels)
            loss = self.criterion(model_out, labels.type_as(model_out))
            losses.append(loss.item())
            if is_train:
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            else:
                mAP.append(self._compute_map(model_out, labels))
                acc.append(self._compute_accuracy(model_out, labels))
                
        if is_train: return np.mean(losses)
        else: return np.mean(losses), np.mean(mAP), np.mean(acc)
    
    
    def train(self, train_dataloader, val_dataloader, save_folder='./checkpoints/', num_epochs=240):
        best_loss = 100
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = self._step(
                train_dataloader,
                is_train=True,
            )
            with torch.no_grad():
                val_loss, mAP, acc = self._step(
                    val_dataloader,
                )
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model, 'classifer.pth')
            print(f'epoch: {epoch}, train_loss: {train_loss}, val_loss: {val_loss}, val_map: {mAP}, val_acc: {acc}')
            self.scheduler.step(val_loss)
            
    def predict(self, filepath, base_transforms):
        image = Image.open(filepath).convert('RGB')
        image = base_transforms(image)
        image = torch.unsqueeze(image, 0)
        pred = self.model.forward(image)
        pred = pred.detach().numpy()
        pred_labels = self._one_hot2labels(pred)[0]
        return pred_labels, pred[0][pred_labels]
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train loop for classification net.')
    parser.add_argument('csv_file', type=str, help='path to csv file that contains dataset info in format:\n1.jpg,0\n2.jpg,1')
    parser.add_argument('root_dir', type=str, help='path to root dir with dataset')
    args = parser.parse_args()
    
    csv_file = args.csv_file
    root_dir = args.root_dir

    train_loader, val_loader, num_classes = create_dataloader(
        csv_file=csv_file,
        root_dir=root_dir,
    )
    model = torchvision.models.resnext50_32x4d(pretrained=True)
    model= FineTuneModel(model, num_classes=num_classes)
    model = ModelInterface(model=model)
    model.train(train_loader, val_loader) 
