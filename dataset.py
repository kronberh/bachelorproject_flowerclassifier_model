import os
import pandas as pd
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset

transform = T.Compose([
    T.Resize(224),
    T.CenterCrop((224, 224)),
    T.ToTensor()
])

class PendingImagesDataset(Dataset):
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join('pending_images', row['path'])
        label = int(row['label'])
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        return image, label
