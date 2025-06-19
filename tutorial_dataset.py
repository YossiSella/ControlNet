import json
import cv2
import numpy as np

from torch.utils.data import Dataset

# Function to read images with unicode paths
# This is necessary because OpenCV does not handle unicode paths directly on Windows.
def imread_unicode(path):
    # Read file manually and decode as image
    stream = open(path, "rb")
    bytes_array = bytearray(stream.read())
    np_array = np.asarray(bytes_array, dtype=np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return img

class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        with open(r"C:\Users\yossi\Documents\Bar Ilan\Master's\תשפה - סמסטר א\מודלים גנרטיביים עמוקים\Final_project\NSTTS\datasets\fill50k\fill50k\prompt.json", 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = imread_unicode(r"C:\Users\yossi\Documents\Bar Ilan\Master's\תשפה - סמסטר א\מודלים גנרטיביים עמוקים\Final_project\NSTTS\datasets\fill50k\fill50k\\" + source_filename)
        target = imread_unicode(r"C:\Users\yossi\Documents\Bar Ilan\Master's\תשפה - סמסטר א\מודלים גנרטיביים עמוקים\Final_project\NSTTS\datasets\fill50k\fill50k\\" + target_filename)

        if source is None or target is None or prompt is None:
            raise ValueError(f"Missing data for index {idx}: source={source_filename}, target={target_filename}, prompt={prompt}")
        
        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

