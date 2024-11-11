import os
import gdown

gdown.download(id='1-rRn4bodjahXQibfC0SqEblzYOaazjEo')
os.makedirs("data/datasets", exist_ok=True)
path_to_zip_file = "small_openwebtext_tok.zip"
directory_to_extract_to = "data/datasets"
import zipfile
with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
  zip_ref.extractall(directory_to_extract_to)
os.remove("small_openwebtext_tok.zip")

gdown.download(id='1PYT0L1rX91EI9BU_K5E2hWXpPSTSyBqC')
os.remove('data/datasets/small_openwebtext_tok/train_index.json')
os.rename("ds_train_index.json", "data/datasets/small_openwebtext_tok/train_index.json")

