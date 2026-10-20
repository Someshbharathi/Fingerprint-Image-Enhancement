# Fingerprint Image Enhancement Using U-Net

A deep learning project that aims to enhance degraded fingerprint images using a U-Net-based image reconstruction model built with PyTorch.

## 📌 Project Overview

Fingerprint images may contain noise, blur, and contrast variations that make ridge patterns difficult to distinguish. This project explores a U-Net architecture to reconstruct fingerprint images from synthetically degraded inputs.

The model is trained using pairs of:
- **Input:** A degraded fingerprint image.
- **Target:** The corresponding original fingerprint image.

The objective is to produce an enhanced image that more closely resembles the original target.

> **Note:** Model performance will be reported after full training and evaluation on the held-out test dataset.

## 🎯 Objectives

- Load and preprocess fingerprint images.
- Generate synthetically degraded fingerprint images.
- Implement a U-Net architecture for image-to-image reconstruction.
- Train the model using PyTorch.
- Evaluate reconstruction quality using image-quality metrics.
- Visualize and compare degraded inputs, enhanced outputs, and original targets.

## 🏗️ Project Structure

```text
Fingerprint-Image-Enhancement/
├── datasets/
│   └── fingerprint_images/
├── outputs/
│   ├── models/
│   ├── enhanced_images/
│   └── evaluation/
├── src/
│   └── dataset.py
├── Fingerprint_Enhancement.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

The exact dataset subfolder structure may vary depending on how the images are organized.

### Directory Description

| Path | Purpose |
|---|---|
| `datasets/` | Fingerprint image dataset |
| `outputs/models/` | Saved model checkpoints |
| `outputs/enhanced_images/` | Generated enhanced images |
| `outputs/evaluation/` | Evaluation metrics and results |
| `src/dataset.py` | Dataset loading, preprocessing, and degradation logic |
| `Fingerprint_Enhancement.ipynb` | Model development, training, and evaluation |
| `requirements.txt` | Python dependencies |

## 🧠 Model Architecture

The project uses a U-Net convolutional neural network.

Its main components are:

- **Encoder:** Extracts spatial features through convolutional blocks and downsampling.
- **Bottleneck:** Learns deeper feature representations.
- **Decoder:** Reconstructs the image through upsampling and convolutional layers.
- **Skip connections:** Transfer encoder features to corresponding decoder stages to preserve spatial details.
- **Sigmoid output:** Produces a single-channel image with pixel values in the range `[0, 1]`.

The model accepts grayscale fingerprint images resized to `256 × 256` and produces an output of the same dimensions.

## 🔄 Workflow

1. Load fingerprint images from the dataset.
2. Convert images to grayscale, resize them to `256 × 256`, and normalize pixel values.
3. Generate degraded inputs using blur, noise, and contrast variations.
4. Use the original images as reconstruction targets.
5. Split the dataset into training, validation, and testing sets.
6. Train the U-Net using a reconstruction loss function and an optimizer.
7. Save the best model checkpoint based on validation performance.
8. Evaluate the trained model on the held-out test set.
9. Compare degraded inputs, enhanced outputs, and original targets.

### Data Degradation

During training, synthetic degradation can include:
- Gaussian blur.
- Random noise.
- Contrast and brightness variations.

Validation and test degradation use fixed settings to support reproducible comparisons.

The results therefore measure reconstruction under the selected synthetic degradation conditions. They do not automatically establish improved fingerprint identification accuracy.

## 🛠️ Technologies Used

- Python
- PyTorch
- NumPy
- OpenCV
- Pillow
- Matplotlib
- scikit-image
- scikit-learn
- Jupyter Notebook

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Someshbharathi/Fingerprint-Image-Enhancement.git
cd Fingerprint-Image-Enhancement
```

### 2. Create a virtual environment

Python 3.11 is the suggested starting point.

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install PyTorch separately if it is not included in the requirements file. Select the appropriate installation command for your operating system and compute platform from the official [PyTorch installation guide](https://pytorch.org/get-started/locally/).

### 4. Verify PyTorch and GPU availability

```python
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

If CUDA is unavailable, verify that PyTorch and the GPU environment are configured correctly.

## 📂 Dataset Preparation

Place the fingerprint images inside the `datasets/` directory, following the structure expected by the dataset-loading code.

Before training, verify that:

- The dataset path is correct.
- Images can be loaded successfully.
- Images are converted to grayscale and normalized to `[0, 1]`.
- The train, validation, and test sets contain the expected number of images.
- Related images from the same fingerprint identity do not cross dataset splits, if identity-wise splitting is intended.

The dataset source, license, and any usage restrictions should be documented here once finalized.

## 🚀 Training

Open `Fingerprint_Enhancement.ipynb` in VS Code or Jupyter Notebook.

1. Activate the project environment.
2. Select the correct Python kernel.
3. Run the notebook cells in order.
4. Verify dataset loading and model output shapes.
5. Run a short training smoke test.
6. Start full training on a compatible GPU.
7. Confirm that the best checkpoint is saved successfully.

The training configuration, including the number of epochs, batch size, learning rate, and early-stopping patience, should be documented alongside the final experiment results.

## 📊 Evaluation

The trained model should be evaluated on the held-out test dataset.

### Suggested Metrics

| Metric | Purpose |
|---|---|
| MSE | Measures average squared pixel-level reconstruction error |
| PSNR | Measures reconstruction quality on a logarithmic scale |
| SSIM | Measures structural similarity between reconstructed and target images |

Compare the degraded input, enhanced output, and original target visually as well as numerically.

Evaluation results should be added after the final checkpoint has been tested. Do not report expected or pilot results as final test performance.

## 📈 Results

This section will contain the results after full training.

Recommended deliverables:
- Training and validation loss curves.
- Final test-set MSE, PSNR, and SSIM.
- Sample visual comparisons.
- Model configuration and training details.
- Limitations and observations.

## ⚠️ Limitations

- Synthetic degradation may not represent every type of real-world fingerprint damage.
- Pixel-level reconstruction metrics do not directly measure fingerprint matching or identification performance.
- Results depend on dataset quality, split methodology, training configuration, and the selected degradation process.
- Further evaluation is needed before drawing conclusions about real-world biometric performance.

## 🔮 Future Improvements

- Evaluate multiple degradation types and severity levels.
- Compare U-Net with alternative restoration architectures.
- Investigate real-world noisy fingerprint datasets.
- Evaluate whether enhancement improves downstream fingerprint matching.
- Optimize model size and inference speed.

## 👥 Contributors

- Rutuja Shinde
- Somesh Bharathi
- Neha Bhaskar

