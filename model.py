import csv
from datetime import datetime, timezone
import os
import shutil
import torch
from PIL import Image
from tqdm import tqdm
from torchvision.models import resnet50
from dataset import PendingImagesDataset, transform

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

model = resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 299)
model.load_state_dict(torch.load('flower_classifier.pth', map_location=device))
model.to(device)
model.eval()

@torch.no_grad()
def classify_image(image_path: str, probability_threshold: int = 0) -> tuple[str, float]:
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)
    output = model(tensor)
    probs = torch.softmax(output, dim=1).squeeze(0)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    top_conf = sorted_probs[0].item()
    min_conf = top_conf - (probability_threshold / 100)
    results = []
    for conf, idx in zip(sorted_probs, sorted_indices):
        if conf.item() >= min_conf:
            results.append((idx.item(), conf.item()))
        else:
            break
    return results

def scan_image(image_path: str, label: int, user_id: int):
    ext = Image.open(image_path).format.lower()
    ts = int(datetime.now(timezone.utc).timestamp())
    filename = f'{user_id}_{ts}.{ext}'
    shutil.copy(image_path, os.path.join('pending_images', filename))
    with open('pending_images_dataset.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([filename, label])

def process_pending_images():
    dataset = PendingImagesDataset('pending_images_dataset.csv')
    if len(dataset) == 0:
        return
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for images, labels in tqdm(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    model.eval()

    torch.save(model.state_dict(), 'flower_classifier.pth')
    model.load_state_dict(torch.load('flower_classifier.pth', map_location=device))
    
    os.remove('pending_images_dataset.csv')
    for filename in os.listdir('pending_images'):
        os.remove(os.path.join('pending_images', filename))
    with open('pending_images_dataset.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'label'])
