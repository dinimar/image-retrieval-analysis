import yaml
import wget
import os
import sys
import zipfile

def unzip(file_path, dir_path='./'):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(dir_path)



def download_datasets(yaml_dir):
    with open(os.path.join(yaml_dir, 'digix.yaml')) as f:
        data = yaml.safe_load(f)
        
        print("Started downloading from", data['train_data'])
        wget.download(data['train_data'], data['train_data_path'], bar=bar_progress)
        print("Finished downloading\r")
        print("Started unzipping", data['train_data_path'])
        unzip(data['train_data_path'])
        print("Finished unzipping\r")
        
        print("Started downloading from", data['test_data'])
        wget.download(data['test_data'], data['test_data_path'], bar=bar_progress)
        print("Finished downloading\r")
        
        print("Started unzipping", data['train_data_path'])
        unzip(data['test_data_path'])
        print("Finished unzipping\r")
        