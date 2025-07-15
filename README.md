# deepHSI
README: Spectral Autoencoder-Decoder Network for Hyperspectral Denoising and Segmentation

project github: https://github.com/jyhuang239/deepHSI.git

Last Updated: July 10, 2025

📁 File Structure </br>
├── auto_encoderdecoder.py         # Autoencoder-decoder model </br>
├── train_HSImodel_on_tifstack.py  # Training script </br>
├── run_HSImodel_on_tifstack.py    # Model prediction, latent-space clustering, and segmentation script </br>
├── HSI_400-1000nm/ </br>
│   └── WHU-LongKou.tif            # Sample hyperspectral image stack </br>
├── model/          </br>
│   └── WHU-LongKou_hsi.pt         # Output trained model </br>

Part 1. Spectral Autoencoder-Decoder Network for Hyperspectral Denoising and Segmentation

## 📌 Purpose

This autoencoder-decoder (AED) neural network is designed specifically for **spectral denoising and segmentation** of hyperspectral imaging (HSI) data. It operates on spectral vectors (e.g., per-pixel spectra from hyperspectral cubes) and learns a compressed latent representation that preserves key spectral features while reducing noise and redundancy.

---

## ⚙️ Functionality

The network processes 1D spectral data using convolutional layers in a symmetric architecture:

- **Encoder**: Applies a series of `Conv1D` layers to reduce the spectral dimension and extract meaningful features.
- **Latent Space**: A fully connected bottleneck transforms the encoded feature map into a compact latent vector of dimension `zdims`.
- **Decoder**: Uses `ConvTranspose1D` layers to reconstruct the spectral signal from the latent space. The architecture ensures that the output spectral length is identical to the input, even with multiple downsampling/upsampling stages.

This network is suitable for:
- Denoising hyperspectral data
- Spectral feature compression and reconstruction
- Serving as a spectral submodule for HSI tasks like classification or unmixing

---

## 🌟 Unique Features of the AED architecture

- **Symmetric Architecture**  
  Encoder and decoder are mirror images in design, enabling structured and consistent mapping between input and output.

- **Dynamic Spectral Shape Tracking**  
  During encoding, the network records the intermediate spectral lengths after each convolutional layer.  
  This tracking allows the decoder to precisely calculate and apply **dynamic `output_padding`** during each deconvolution to recover the original spectral resolution.

- **Automatic Padding Estimation**  
  The `_calculate_output_padding()` function ensures perfect inverse mapping from latent space to original spectral length — eliminating common size mismatch errors.

- **Fully Connected Latent Bottleneck**  
  Uses a `Linear` layer to project the flattened encoder output to a compact latent space and vice versa — enabling flexibility in latent dimensionality and compression strength.

- **Tanh Activation & Batch Normalization**  
  Ensures bounded output in the range [-1, 1], which is well-suited for normalized spectral data.  
  Batch normalization accelerates training and enhances stability.

---

## 🧪 Inputs and Outputs

- **Input**: 1D spectral tensor of shape `(batch_size, 1, n_frames)`
- **Output**: Reconstructed spectral tensor of the same shape `(batch_size, 1, n_frames)`
- **Latent Vector**: Intermediate output `z` of shape `(batch_size, zdims)`

---

## 🧩 Customization

- Set `zdims` to define the size of the latent space.
- Set `n_frames` to match the input spectral length (number of bands).

Example initialization:
```python
model = Autoencoder(zdims=32, n_frames=128)


# 🧠 Part II. Autoencoder-Decoder Training for Hyperspectral Image Stack

This repository provides a training script for learning a **spectral autoencoder-decoder (AED)** from a **TIFF hyperspectral image stack**. The model extracts latent spectral features and clusters them using spatially-aware Gaussian Mixture Models (GMMs). It is optimized with a rich composite loss function that enhances both global and local spectral fidelity.

---

## 🚀 Functionality Overview

### ✅ 1. **Data Handling**
- Loads a `.tif` hyperspectral image stack (e.g., `WHU-LongKou.tif`).
- Reshapes it from shape `(n_frames, dim_x, dim_y)` to `(dim_x, dim_y, n_frames)` for spectral processing.
- Resamples each pixel’s spectrum to a uniform length (default: 512 bands).
- Normalizes the data and flattens it for input into a 1D CNN autoencoder.

### ✅ 2. **Autoencoder Training**
- Uses a symmetric 1D convolutional autoencoder (see `auto_encoderdecoder.py`) that:
  - Encodes spectra into latent vectors.
  - Decodes while preserving the original spectral length using dynamic padding.
- The training uses `90%` of the pixels; `10%` is used for validation.

### ✅ 3. **Composite Loss Function**
The total loss is a weighted sum of:
- **MSE Loss** – Enforces global similarity between input and output spectra.
- **Spectral Angle Mapper (SAM)** – Captures angular spectral variation, sensitive to shape, not magnitude.
- **Gradient Loss** – Preserves spectral edges, enhancing contrast and sharp transitions.

### ✅ 4. **Output & Visualization**
- Training and test losses are plotted (linear and log scale).
  - Best-performing model is saved as a .pt file.
  - Spectral slices are visualized for quick inspection.

## ✨ Unique Features
- Dynamic Reconstruction: Decoder dynamically estimates padding using encoder’s recorded sizes for exact spectral shape recovery.
- Composite Spectral Loss: Combined loss offers a holistic approach, balancing magnitude, shape, and edge preservation.
- Tiff Stack Support: Compatible with multi-frame .tif hyperspectral images (commonly used in biomedical and remote sensing domains).

## 🧪 Example Usage

python train_HSImodel_on_tifstack.py
Make sure the input TIFF file is located at:
./HSI_400-1000nm/WHU-LongKou.tif

## 📦 Output
- WHU-LongKou_hsi.pt: Trained model file
- train_error & test_error plots: Saved as PNG or shown via matplotlib
- Visualizations of input slices and loss convergence


# 🧩 Part III. Latent-Space Image Segmentation of Hyperspectral Data using Auto-encoder-decoder and GMM

This script performs **semantic segmentation** on hyperspectral images by projecting each pixel’s spectrum into a **latent space** using a trained auto-encoder-decoder, and clustering the latent representations using a **Gaussian Mixture Model (GMM)**. This segmentation process uniquely leverages both **spectral and spatial features** to group pixels with similar spectral signatures and spatial coherence.

---

## 🔍 Purpose

The goal of this pipeline is to:
- Predict latent representations of pixel-wise spectra from a trained AED model.
- Perform **unsupervised image segmentation** based on the spectral features and spatial coherence.
- Visualize and interpret the results through both spatially-clustered maps and representative cluster spectra.

---

## 🧠 Functional Workflow

### 🔹 1. Data Loading and Preprocessing
- Input: TIFF hyperspectral stack `(n_frames, height, width)`
- Normalization: Mean-centered and range-scaled
- Resampling: Each spectrum is interpolated to a common spectral resolution (`target_len = 512`)
- Reshaping: Converted to shape `(n_pixels, 1, n_bands)` for model input

### 🔹 2. Latent Space Projection
- Loads the pretrained autoencoder model
- Encodes all spectra into latent vectors `z` of dimension `zdims`
- Extracted latent vectors form the basis for clustering

### 🔹 3. Spatially-Aware GMM Clustering
- The latent vectors are combined with normalized pixel coordinates `(x, y)`
- A **Gaussian Mixture Model** clusters the augmented feature space
- This enables clusters that reflect both **spectral similarity** and **spatial proximity**

### 🔹 4. Cluster Analysis and Visualization
- Reconstructs representative spectra for each cluster by decoding the GMM mean vectors
- Displays:
  - Segmented cluster map
  - Comparison of AED output and input spectral slices
  - Cluster-wise spectra from decoded latent centers

---

## 🌟 Unique Features

- ✅ **Spectral + Spatial Clustering**: Clustering uses both latent spectra and pixel coordinates for semantically meaningful grouping.
- ✅ **Cluster Spectrum Decoding**: Decodes cluster center vectors to visualize their corresponding spectra for constitutive identification.
- ✅ **Fully Unsupervised**: No labels required; clustering is driven entirely by learned features.
- ✅ **Reconstruction Comparison**: Plots spectra from the input and output at specific pixel locations for model validation.
- ✅ **Flexible GMM Clustering**: Easily change the number of clusters (`n_clusters`) to match complexity.

---

## 📊 Outputs

- **Segmented image** with color-coded clusters
- **Spectral plot** of a selected pixel (input vs AED output)
- **Average spectra** for each cluster (decoded from latent space)
- **Reconstructed output image** (for visual inspection)

## 🧪 Sample Workflow

```bash
python run_HSImodel_on_tifstack.py
Ensure the trained model WHU-LongKou_hsi.pt is located in:
./model/

And the input .tif file (e.g., WHU-LongKou.tif) is located in:
./HSI_400-1000nm/

## 📈 Example Visualizations
- Clustered segmentation map overlaid with color-coded labels
- Model reconstruction vs original spectral slice
- Spectral plot comparing AED output and input for a selected pixel
- Representative cluster spectra decoded from GMM means
