import os
import shutil
import random
from pathlib import Path

random.seed(42)

DATA_DIR    = "data"
CLASSES     = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
TRAIN_DIR   = os.path.join(DATA_DIR, "Training")
TEST_DIR    = os.path.join(DATA_DIR, "Testing")
BACKUP_DIR  = os.path.join(DATA_DIR, "_backup")


if not os.path.exists(BACKUP_DIR):
    print("XATO: Backup topilmadi! Avval asl ma'lumotlarni tiklang.")
    exit(1)
else:
    print(f"Backup topildi: {BACKUP_DIR}")


print("\nBackup dan rasmlar o'qilmoqda...")

backup_train = os.path.join(BACKUP_DIR, "Training")
backup_test  = os.path.join(BACKUP_DIR, "Testing")

stats = {}

for cls in CLASSES:
    all_images = []

    for src_dir in [
        os.path.join(backup_train, cls),
        os.path.join(backup_test,  cls),
    ]:
        if os.path.exists(src_dir):
            for f in os.listdir(src_dir):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    full_path = os.path.join(src_dir, f)
                    all_images.append(full_path)

    random.shuffle(all_images)
    total   = len(all_images)
    n_train = int(total * 0.80)
    n_test  = total - n_train

    train_images = all_images[:n_train]
    test_images  = all_images[n_train:]

    # Papkalarni yangilash
    new_train_cls = os.path.join(TRAIN_DIR, cls)
    new_test_cls  = os.path.join(TEST_DIR,  cls)
    shutil.rmtree(new_train_cls, ignore_errors=True)
    shutil.rmtree(new_test_cls,  ignore_errors=True)
    os.makedirs(new_train_cls, exist_ok=True)
    os.makedirs(new_test_cls,  exist_ok=True)

    # Ko'chirish — nom to'qnashuvi bo'lsa nomer qo'shish
    def safe_copy(src, dst_dir):
        fname = os.path.basename(src)
        dst   = os.path.join(dst_dir, fname)
        if os.path.exists(dst):
            name, ext = os.path.splitext(fname)
            dst = os.path.join(dst_dir, f"{name}_{random.randint(10000,99999)}{ext}")
        shutil.copy2(src, dst)

    for src in train_images:
        safe_copy(src, new_train_cls)
    for src in test_images:
        safe_copy(src, new_test_cls)

    stats[cls] = {"total": total, "train": n_train, "test": n_test}
    print(f"  {cls:22s}: jami={total:4d} → train={n_train:4d} | test={n_test:3d}")


print("\nYangi taqsimot:")
print(f"  {'Sinf':22s} {'Jami':>6} {'Train':>7} {'Test':>6}")
print("  " + "-" * 44)
for cls, s in stats.items():
    print(f"  {cls:22s} {s['total']:6d} {s['train']:7d} {s['test']:6d}")

total_all   = sum(s["total"] for s in stats.values())
total_train = sum(s["train"] for s in stats.values())
total_test  = sum(s["test"]  for s in stats.values())
print("  " + "-" * 44)
print(f"  {'JAMI':22s} {total_all:6d} {total_train:7d} {total_test:6d}")
print(f"\n  Train: {total_train/total_all*100:.1f}%  |  Test: {total_test/total_all*100:.1f}%")
print("\nTayyor! Endi brain_tumor_classification.py ni ishga tushiring.")