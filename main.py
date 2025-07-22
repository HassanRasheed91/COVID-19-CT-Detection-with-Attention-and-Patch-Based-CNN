import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from patch_extractor import extract_patches
from ensemble import PatchAttentionEnsemble

class CTScanDataset(Dataset):
    def __init__(self, root_dir, label_map):
        self.samples = []
        self.label_map = label_map
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        for label in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, label)
            for fname in os.listdir(class_dir):
                self.samples.append((os.path.join(class_dir, fname), label_map[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        patches = extract_patches(image, grid_size=3)
        patches = torch.stack([self.transform(p) for p in patches])
        return patches, label

train_dataset = CTScanDataset("./train", label_map={"COVID": 1, "Non-COVID": 0})
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

model = PatchAttentionEnsemble().cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()
    running_loss = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {running_loss/len(train_loader):.4f}")
