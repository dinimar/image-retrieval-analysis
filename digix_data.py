import yaml
import wget
import os
import sys

# create this bar_progress method which is invoked automatically from wget
def bar_progress(current, total, width=80):
    progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
    # Don't use print() as it will print in new line every time.
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


def download_datasets(yaml_dir):
    with open(os.path.join(yaml_dir, 'digix.yaml')) as f:
        data = yaml.safe_load(f)
        
        print("Started downloading from", data['train_data'])
        wget.download(data['train_data'], bar=bar_progress)
        print("Started downloading from", data['test_data'])
        wget.download(data['test_data'], bar=bar_progress)