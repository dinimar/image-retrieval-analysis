import yaml
import wget

def download_datasets():
    with open('digix.yaml') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        
        wget.download(data['train_data'])
        wget.download(data['test_data'])