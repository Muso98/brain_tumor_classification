import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


DATA_DIR   = "data/Testing"
CKPT_PATH  = "checkpoints/resnet50_best.pt"
SAVE_PATH  = "gradcam_results.png"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
CLASS_LABELS = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"]

random.seed(42)


model = models.resnet50(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(2048, 4)
)
ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt)
model = model.to(DEVICE)
model.eval()
print(f"Model yuklandi: {CKPT_PATH}")


preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def get_sample_image(class_name, n=5):
    """Har sinfdan to'g'ri klassifikatsiya qilingan rasmni topadi."""
    class_dir = os.path.join(DATA_DIR, class_name)
    files = [f for f in os.listdir(class_dir)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(files)

    class_idx = CLASS_NAMES.index(class_name)

    for fname in files[:20]:
        path = os.path.join(class_dir, fname)
        img_pil = Image.open(path).convert("RGB")
        img_tensor = preprocess(img_pil).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(img_tensor)
            pred = output.argmax(1).item()
            conf = torch.softmax(output, dim=1)[0][pred].item()

        if pred == class_idx and conf > 0.85:
            return path, img_pil, class_idx, conf

    path = os.path.join(class_dir, files[0])
    img_pil = Image.open(path).convert("RGB")
    img_tensor = preprocess(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(img_tensor)
        pred = output.argmax(1).item()
        conf = torch.softmax(output, dim=1)[0][pred].item()
    return path, img_pil, pred, conf


target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers)


fig, axes = plt.subplots(4, 3, figsize=(12, 16))
fig.suptitle("Grad-CAM Visualization — ResNet-50\nModel Attention on Brain Tumor MRI",
             fontsize=14, fontweight="bold", y=0.98)

col_titles = ["Original MRI", "Grad-CAM Heatmap", "Overlay"]
for col, title in enumerate(col_titles):
    axes[0][col].set_title(title, fontsize=12, fontweight="bold", pad=10)

colors = ["#E05A4E", "#028090", "#2ECC71", "#9B59B6"]

for row, class_name in enumerate(CLASS_NAMES):
    print(f"  Processing: {CLASS_LABELS[row]}...")

    path, img_pil, pred_idx, conf = get_sample_image(class_name)

    img_resized = img_pil.resize((224, 224))
    img_np = np.array(img_resized).astype(np.float32) / 255.0

    img_tensor = preprocess(img_pil).unsqueeze(0).to(DEVICE)

    targets = [ClassifierOutputTarget(CLASS_NAMES.index(class_name))]
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0]

    overlay = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    axes[row][0].imshow(img_resized, cmap="gray")
    axes[row][0].set_ylabel(
        CLASS_LABELS[row],
        fontsize=11, fontweight="bold",
        color=colors[row], labelpad=10
    )

    axes[row][1].imshow(grayscale_cam, cmap="jet")

    axes[row][2].imshow(overlay)

    pred_name = CLASS_LABELS[pred_idx]
    match = pred_idx == CLASS_NAMES.index(class_name)
    label_color = "#27AE60" if match else "#E74C3C"
    label_symbol = "✓" if match else "✗"
    axes[row][2].set_xlabel(
        f"{label_symbol} Pred: {pred_name}\nConf: {conf:.1%}",
        fontsize=9, color=label_color, fontweight="bold"
    )

    for col in range(3):
        axes[row][col].set_xticks([])
        axes[row][col].set_yticks([])
        for spine in axes[row][col].spines.values():
            spine.set_edgecolor(colors[row])
            spine.set_linewidth(2)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(SAVE_PATH, dpi=150, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.show()
print(f"\nSaqlandi: {SAVE_PATH}")