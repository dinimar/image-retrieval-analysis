import yaml
import wget
import os

def download_datasets(yaml_dir):
    with open(os.path.join(yaml_dir, 'digix.yaml')) as f:
        data = yaml.safe_load(f)
        
        wget.download(data['train_data'])
        wget.download(data['test_data'])