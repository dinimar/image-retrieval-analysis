import yaml
import wget
import os

def download_datasets(yaml_path):
    with open(os.path.join(yaml_path, 'digix.yaml')) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        
        wget.download(data['train_data'])
        wget.download(data['test_data'])