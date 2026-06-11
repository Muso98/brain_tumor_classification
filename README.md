# 🧠 Brain Tumor Classification using PyTorch

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A deep learning project for classifying brain MRI scans into four distinct categories using an ensemble of advanced Convolutional Neural Networks (CNNs). This project implements a robust pipeline designed to handle medical imaging challenges such as class imbalance and feature extraction subtlety.

## 📌 Features
* **Multi-Model Ensemble**: Combines **ResNet50** and **EfficientNet-B3** using weighted soft voting for high accuracy.
* **Specialized Binary Classifier**: Incorporates a dedicated ResNet50-based binary classifier specifically fine-tuned to distinguish tricky *Glioma* cases.
* **Test Time Augmentation (TTA)**: Applies multiple augmentations during inference to improve prediction stability and confidence.
* **Advanced Loss Functions**: Utilizes **Focal Loss** to focus learning on hard-to-classify examples.
* **Class Imbalance Handling**: Uses `WeightedRandomSampler` to ensure uniform exposure to all classes during training.
* **Explainability (XAI)**: Includes **Grad-CAM** implementations to visualize model focus areas, critical for medical diagnosis transparency.

## 🗂️ Dataset Classes
The models classify MRI images into 4 categories:
1. `glioma_tumor`
2. `meningioma_tumor`
3. `no_tumor`
4. `pituitary_tumor`

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Muso98/brain_tumor_classification.git
   cd brain_tumor_classification
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Preparation:**
   Ensure your data is placed in the `data/` directory with the following structure:
   ```text
   data/
   ├── Training/
   │   ├── glioma_tumor/
   │   ├── meningioma_tumor/
   │   ├── no_tumor/
   │   └── pituitary_tumor/
   └── Testing/
       ├── glioma_tumor/
       └── ...
   ```

## 🚀 Usage

### Training the Models
To train the individual models and the binary classifier, and eventually evaluate the ensemble:
```bash
python brain_tumor_classification.py
```
*Note: Checkpoints will be automatically saved to the `checkpoints/` directory as `*_best.pt` files.*

### Visualizing Results (Grad-CAM)
To generate attention maps (Grad-CAM) showing what the model looked at to make its prediction:
```bash
python gradcam_viz.py
```

## 📊 Results & Performance
The project automatically generates comprehensive plots:
* **Training Curves** (`plots_training.png`): Loss, Accuracy, and F1-Score trends over epochs.
* **Confusion Matrices** (`plots_cm_*.png`): Detailed class-wise accuracy.
* **Comparison Plot** (`plots_comparison.png`): Bar charts comparing standalone ResNet50, EfficientNet-B3, and the final Ensemble.

## 📝 License
This project is open-source and available under the MIT License.
