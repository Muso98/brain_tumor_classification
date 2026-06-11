import os
import time
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score
)


CONFIG = {
    "data_dir":         "data",
    "batch_size":       32,
    "num_epochs":       25,
    "lr":               1e-4,
    "weight_decay":     1e-4,
    "img_size":         224,
    "num_classes":      4,
    "device":           "cuda" if torch.cuda.is_available() else "cpu",
    "seed":             42,
    "save_dir":         "checkpoints",
    "binary_epochs":    20,
    "binary_lr":        1e-4,
    "ensemble_w":       {"resnet50": 0.55, "efficientnet_b3": 0.45},
    "glioma_threshold": 0.55,
}

CLASS_NAMES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
GLIOMA_IDX  = 0

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
os.makedirs(CONFIG["save_dir"], exist_ok=True)

print(f"Device : {CONFIG['device']}")
print(f"Classes: {CLASS_NAMES}")


train_transforms = transforms.Compose([
    transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

tta_transforms = [
    val_transforms,
    transforms.Compose([
        transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize((int(CONFIG["img_size"] * 1.1), int(CONFIG["img_size"] * 1.1))),
        transforms.CenterCrop(CONFIG["img_size"]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
        transforms.RandomRotation(degrees=(10, 10)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
]


def get_dataloaders(data_dir, batch_size):
    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "Training"), transform=train_transforms
    )
    test_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "Testing"), transform=val_transforms
    )

    class_counts  = np.bincount(train_dataset.targets)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[t] for t in train_dataset.targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=sampler, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False,  num_workers=2, pin_memory=True)

    print(f"\nTraining samples : {len(train_dataset)}")
    print(f"Testing  samples : {len(test_dataset)}")
    print(f"Class distribution: {dict(zip(CLASS_NAMES, class_counts))}")

    return train_loader, test_loader, train_dataset, test_dataset


class BinaryGliomaDataset(Dataset):
    def __init__(self, base_dataset, transform=None):
        self.samples   = base_dataset.samples
        self.transform = transform or base_dataset.transform
        self.loader    = base_dataset.loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, orig_label = self.samples[idx]
        img = self.loader(path)
        img = self.transform(img)
        return img, (1 if orig_label == GLIOMA_IDX else 0)


def get_binary_loaders(train_dataset, test_dataset, batch_size):
    bin_train = BinaryGliomaDataset(train_dataset, train_transforms)
    bin_test  = BinaryGliomaDataset(test_dataset,  val_transforms)

    labels   = [1 if s[1] == GLIOMA_IDX else 0 for s in train_dataset.samples]
    counts   = np.bincount(labels)
    w        = 1.0 / counts
    sample_w = [w[l] for l in labels]
    sampler  = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

    train_loader = DataLoader(bin_train, batch_size=batch_size,
                              sampler=sampler, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(bin_test,  batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    glioma_count = sum(1 for s in train_dataset.samples if s[1] == GLIOMA_IDX)
    print(f"\nBinary dataset — Glioma: {glioma_count} | Rest: {len(train_dataset)-glioma_count}")

    return train_loader, test_loader


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction
        self.ce        = nn.CrossEntropyLoss(reduction="none")

    def forward(self, inputs, targets):
        ce_loss    = self.ce(inputs, targets)
        pt         = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == "mean" else focal_loss.sum()


def build_model(model_name, num_classes, device):
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.fc.in_features, num_classes))

    elif model_name == "efficientnet_b3":
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        in_f  = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_f, num_classes))

    elif model_name == "binary_resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, 2))

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model.to(device)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        correct += model(images).detach().argmax(1).eq(labels).sum().item()
        total   += labels.size(0)
    return running_loss / total, correct / total


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total   += labels.size(0)
    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Eval ", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total   += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return running_loss / total, correct / total, f1, all_preds, all_labels


def train_model(model_name, train_loader, test_loader, num_epochs, lr, config, tag=""):
    print(f"\n{'='*55}")
    print(f"  Training: {model_name.upper()} {tag}")
    print(f"{'='*55}")

    device    = config["device"]
    num_cls   = 2 if model_name == "binary_resnet50" else config["num_classes"]
    model     = build_model(model_name, num_cls, device)
    criterion = FocalLoss(alpha=1.0, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_f1, best_weights = 0.0, None

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        for k, v in zip(["train_loss","train_acc","val_loss","val_acc","val_f1"],
                        [train_loss, train_acc, val_loss, val_acc, val_f1]):
            history[k].append(v)

        print(f"  Epoch {epoch:02d}/{num_epochs} | "
              f"Loss {train_loss:.4f}/{val_loss:.4f} | "
              f"Acc {train_acc:.3f}/{val_acc:.3f} | "
              f"F1 {val_f1:.3f} | {time.time()-t0:.1f}s")

        if val_f1 > best_f1:
            best_f1      = val_f1
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, os.path.join(config["save_dir"], f"{model_name}_best.pt"))
            print(f"    ✓ New best F1: {best_f1:.4f} — saved")

    model.load_state_dict(best_weights)
    return model, history


def predict_ensemble_with_binary(models_dict, binary_model, test_dataset, config):
    device    = config["device"]
    threshold = config["glioma_threshold"]
    w         = config["ensemble_w"]

    for m in list(models_dict.values()) + [binary_model]:
        m.eval()

    all_preds, all_labels = [], []
    glioma_overrides = 0

    print("\n  Running ensemble + binary inference (TTA x5)...")

    for idx in tqdm(range(len(test_dataset)), desc="  Inference"):
        path, true_label = test_dataset.samples[idx]
        img_pil = test_dataset.loader(path)

        # Ensemble soft voting with TTA
        ensemble_probs = np.zeros(config["num_classes"])
        binary_probs   = np.zeros(2)

        for t in tta_transforms:
            img_t = t(img_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                for mname, model in models_dict.items():
                    p = torch.softmax(model(img_t), dim=1).cpu().numpy()[0]
                    ensemble_probs += w[mname] * p
                bp = torch.softmax(binary_model(img_t), dim=1).cpu().numpy()[0]
                binary_probs += bp

        ensemble_probs /= len(tta_transforms)
        binary_probs   /= len(tta_transforms)

        # Final decision
        if binary_probs[1] >= threshold:
            final_pred = GLIOMA_IDX
            glioma_overrides += 1
        else:
            final_pred = int(np.argmax(ensemble_probs))

        all_preds.append(final_pred)
        all_labels.append(true_label)

    print(f"  Glioma binary overrides: {glioma_overrides}/{len(test_dataset)}")
    return all_preds, all_labels


def plot_training_curves(histories, names, save_path="plots_training.png"):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    colors = {"resnet50": "#028090", "efficientnet_b3": "#E05A4E", "binary_resnet50": "#8B5CF6"}
    metrics = [("val_loss","Validation Loss"),("val_acc","Validation Accuracy"),("val_f1","Validation Weighted F1")]
    for ax, (key, title) in zip(axes, metrics):
        for name, hist in zip(names, histories):
            ax.plot(hist[key], label=name, color=colors.get(name, "gray"), linewidth=2)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")


def plot_confusion_matrix(labels, preds, class_names, title, save_path):
    cm     = confusion_matrix(labels, preds)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, ax=ax, cbar_kws={"label": "%"})
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")


def plot_comparison(individual_results, ensemble_result, save_path="plots_comparison.png"):
    all_results = {**individual_results, "ensemble_final": ensemble_result}
    names = list(all_results.keys())
    accs  = [all_results[n]["accuracy"] for n in names]
    f1s   = [all_results[n]["f1"]       for n in names]
    x     = np.arange(len(names))
    width = 0.35
    colors_acc = ["#028090", "#E05A4E", "#8B5CF6"]
    colors_f1  = ["#02C39A", "#F4A261", "#A78BFA"]
    fig, ax = plt.subplots(figsize=(9, 4))
    bars1 = ax.bar(x - width/2, accs, width, label="Accuracy",    color=colors_acc)
    bars2 = ax.bar(x + width/2, f1s,  width, label="Weighted F1", color=colors_f1)
    for bar in bars1 + bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_title("Model Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")


def main():
    train_loader, test_loader, train_dataset, test_dataset = get_dataloaders(
        CONFIG["data_dir"], CONFIG["batch_size"]
    )

    # 1. ResNet50
    resnet, hist_resnet = train_model(
        "resnet50", train_loader, test_loader,
        CONFIG["num_epochs"], CONFIG["lr"], CONFIG, tag="(Ensemble)"
    )

    # 2. EfficientNet-B3
    effnet, hist_effnet = train_model(
        "efficientnet_b3", train_loader, test_loader,
        CONFIG["num_epochs"], CONFIG["lr"], CONFIG, tag="(Ensemble)"
    )

    # 3. Binary Glioma Classifier
    bin_train_loader, bin_test_loader = get_binary_loaders(
        train_dataset, test_dataset, CONFIG["batch_size"]
    )
    binary_model, hist_binary = train_model(
        "binary_resnet50", bin_train_loader, bin_test_loader,
        CONFIG["binary_epochs"], CONFIG["binary_lr"], CONFIG, tag="(Glioma Binary)"
    )

    # Training curves
    plot_training_curves(
        [hist_resnet, hist_effnet, hist_binary],
        ["resnet50", "efficientnet_b3", "binary_resnet50"]
    )

    # Individual results
    criterion = FocalLoss(alpha=1.0, gamma=2.0)
    individual_results = {}
    for mname, model in [("resnet50", resnet), ("efficientnet_b3", effnet)]:
        _, _, _, preds, labels = evaluate(model, test_loader, criterion, CONFIG["device"])
        acc = accuracy_score(labels, preds)
        f1  = f1_score(labels, preds, average="weighted")
        individual_results[mname] = {"accuracy": acc, "f1": f1}
        print(f"\n  Classification Report — {mname}")
        print(classification_report(labels, preds, target_names=CLASS_NAMES))
        plot_confusion_matrix(labels, preds, CLASS_NAMES,
                              title=f"Confusion Matrix — {mname}",
                              save_path=f"plots_cm_{mname}.png")

    # Ensemble + Binary final
    models_dict = {"resnet50": resnet, "efficientnet_b3": effnet}
    final_preds, final_labels = predict_ensemble_with_binary(
        models_dict, binary_model, test_dataset, CONFIG
    )
    final_acc = accuracy_score(final_labels, final_preds)
    final_f1  = f1_score(final_labels, final_preds, average="weighted")
    ensemble_result = {"accuracy": final_acc, "f1": final_f1}

    print(f"\n  Classification Report — ENSEMBLE FINAL")
    print(classification_report(final_labels, final_preds, target_names=CLASS_NAMES))
    plot_confusion_matrix(final_labels, final_preds, CLASS_NAMES,
                          title="Confusion Matrix — Ensemble Final",
                          save_path="results/plots_cm_ensemble_final.png")

    plot_comparison(individual_results, ensemble_result)

    print("\n" + "="*50)
    print("  FINAL SUMMARY")
    print("="*50)
    all_res = {**individual_results, "ensemble_final": ensemble_result}
    df = pd.DataFrame(all_res).T
    df.index.name = "Model"
    df.columns    = ["Accuracy", "Weighted F1"]
    print(df.round(4).to_string())
    print("="*50)
    winner = max(all_res, key=lambda k: all_res[k]["f1"])
    print(f"\n  Best: {winner.upper()} — F1 = {all_res[winner]['f1']:.4f}")


if __name__ == "__main__":
    main()