import torch
from torchvision import datasets, transforms, models
import torch.nn as nn
import os
from collections import Counter

device = "cuda" if torch.cuda.is_available() else "cpu"

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test_dataset = datasets.ImageFolder("data/Testing", transform=val_transforms)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

model = models.resnet50(weights=None)
in_features = model.fc.in_features  # 2048
model.fc = nn.Sequential(
    nn.BatchNorm1d(in_features),  # fc.0
    nn.Dropout(0.3),               # fc.1
    nn.Linear(in_features, 256),  # fc.2
    nn.ReLU(),                     # fc.3
    nn.Dropout(0.2),               # fc.4
    nn.Linear(256, 4)             # fc.5
)

ckpt = torch.load("checkpoints/resnet50_best.pt", map_location=device, weights_only=False)
model.load_state_dict(ckpt)
model = model.to(device)
model.eval()
print("Model yuklandi!")

CLASS_NAMES = test_dataset.classes
glioma_idx = CLASS_NAMES.index("glioma_tumor")

wrong = []
with torch.no_grad():
    for i, (img, label) in enumerate(test_loader):
        if label.item() != glioma_idx:
            continue
        out = model(img.to(device))
        pred = out.argmax(1).item()
        if pred != glioma_idx:
            path = test_dataset.samples[i][0]
            wrong.append((os.path.basename(path), CLASS_NAMES[pred]))

print(f"\nXato klassifikatsiya: {len(wrong)}/100 glioma rasmi")
print("Qaysi sinfga o'tkazilgan:")
print(Counter([x[1] for x in wrong]))
print("\nBirinchi 10 ta xato fayl:")
for name, pred in wrong[:10]:
    print(f"  {name} → {pred}")