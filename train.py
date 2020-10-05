import argparse
from dataset import create_dataloader
from model import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train loop for classification net.')
    parser.add_argument('csv_file', type=str, help='path to csv file that contains dataset info in format:\n1.jpg,0\n2.jpg,1')
    parser.add_argument('root_dir', type=str, help='path to root dir with dataset')
    parser.add_argument('num_epochs', type=int, default=4, help='number of epochs that will be trained')
    parser.add_argument('--path_to_model', type=str, help='path to pretrained model')
    args = parser.parse_args()


    train_loader, val_loader, num_classes = create_dataloader(
        csv_file=args.csv_file,
        root_dir=args.root_dir
    )
    if args.path_to_model:
        model = torch.load(args.path_to_model)
    else:
        model = torchvision.models.resnext50_32x4d(pretrained=True)
        model= FineTuneModel(model, num_classes=num_classes)
    model = ModelInterface(model=model)
    model.train(train_loader, val_loader, num_epochs=args.num_epochs) 
