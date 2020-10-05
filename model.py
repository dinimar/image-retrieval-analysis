import torchvision
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import average_precision_score, accuracy_score
import torch
from PIL import Image
import numpy as np

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


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
        # Sets up a timestamped log directory.
        # logdir = "logs/train_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        # Writer will output to ./runs/ directory by default
        self.writer = SummaryWriter()
        
        self.model = model.to(device)
        self.unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
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
        step_num = 0
        for batch in data_loader:
            imgs, labels = batch.values()
            imgs, labels= (
                imgs.to(self.device),
                labels.to(self.device),
            )

            # Print images in TensorBoard
            img_batch = imgs
            for x in range(len(img_batch)):
                img_batch[x] = self.unorm(img_batch[x])
            self.writer.add_images('image_batch', img_batch, step_num)
            step_num = step_num + 1

            model_out = self.model.forward(imgs)
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
    
    
    def train(self, train_dataloader, val_dataloader, save_folder='./checkpoints/', num_epochs=3):
        try:
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

                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('mAP/val', mAP, epoch)
                self.writer.add_scalar('Accuracy/val', acc, epoch)

                print(f'epoch: {epoch}, train_loss: {train_loss}, val_loss: {val_loss}, val_map: {mAP}, val_acc: {acc}')
                self.scheduler.step(val_loss)
        finally:
            # Save results
            self.writer.close()
            
    def predict(self, filepath, base_transforms):
        image = Image.open(filepath).convert('RGB')
        image = base_transforms(image)
        image = torch.unsqueeze(image, 0)
        pred = self.model.forward(image)
        pred = pred.detach().numpy()
        pred_labels = self._one_hot2labels(pred)[0]
        return pred_labels, pred[0][pred_labels]
        

