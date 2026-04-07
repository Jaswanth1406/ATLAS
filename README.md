<div align="center">

# 🗺️ ATLAS

### Adaptive Thresholding with Language-Augmented Sensing

**Mapping Roads with Intelligence**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-15-000000?style=for-the-badge&logo=next.js&logoColor=white)](https://nextjs.org/)
[![License](https://img.shields.io/badge/License-MIT-F59E0B?style=for-the-badge)](LICENSE)

*Full-stack AI road segmentation platform — UNet deep learning + 5 classical methods, served via FastAPI with an interactive Next.js web interface.*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Pages & Features](#-pages--features)
- [API Reference](#-api-reference)
- [Model Details](#-model-details)
- [Segmentation Methods](#-segmentation-methods)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Team](#-team)
- [License](#-license)

---

## 🎯 Overview

### The Problem

Traditional road segmentation struggles with real-world conditions — shadows, uneven lighting, night scenes, and complex urban backgrounds.

### Our Solution

**ATLAS** is a full-stack road segmentation platform combining deep learning with classical computer vision:

- **UNet Model** (ResNet34 encoder) achieving **85%+ IoU**
- **5 Classical Methods** — Otsu, Adaptive Mean/Gaussian, Sauvola, Niblack
- **FastAPI Backend** — 7 API endpoints for inference, comparison, pipeline visualization, batch processing, and annotation
- **Next.js Frontend** — Premium dark-themed UI with interactive segmentation, method comparison, batch processing, and ground truth annotation

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **Deep Learning Model** | UNet + ResNet34 achieving 85%+ IoU, ~15ms GPU inference |
| ⚔️ **Method Comparison** | Side-by-side comparison of UNet vs 5 classical thresholding methods |
| 🌡️ **Confidence Heatmap** | Color-coded probability map showing model certainty per pixel |
| 🔬 **Pipeline Visualizer** | Step-by-step view: resize → normalize → inference → sigmoid → mask → overlay |
| 🎚️ **Threshold Grid** | See results at 6 different thresholds simultaneously |
| ✏️ **Ground Truth Annotation** | Draw road masks on a canvas, compute IoU & Dice vs model prediction |
| 📦 **Batch Processing** | Upload multiple images, get results + downloadable CSV report |
| 💾 **Download Results** | Export mask, overlay, heatmap as PNG or full ZIP package |
| 🕘 **Session History** | Gallery of past predictions — click to reload any result |
| 🌐 **REST API** | 7 FastAPI endpoints with OpenAPI docs at `/docs` |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          ATLAS Architecture                         │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────┐     ┌───────────────┐     ┌─────────────────────┐     │
│  │ Next.js  │────▶│   FastAPI      │────▶│  UNet Model         │     │
│  │ Frontend │◀────│   Backend      │◀────│  (TorchScript)      │     │
│  │ :3000    │     │   :8000        │     │  + Classical Methods │     │
│  └──────────┘     └───────────────┘     └─────────────────────┘     │
│                                                                      │
│  5 Pages:          7 Endpoints:          UNet + 5 Classical:         │
│  Home, Segment,    /predict, /compare,   Otsu, Adaptive Mean,       │
│  Compare, Batch,   /pipeline, /download  Adaptive Gaussian,         │
│  About             /threshold-grid,      Sauvola, Niblack           │
│                    /compute-metrics,                                  │
│                    /predict/batch                                     │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Next.js 15, TypeScript, CSS | 5-page web UI with interactive tools |
| **Backend** | FastAPI, Uvicorn | REST API with 7 endpoints |
| **DL Model** | PyTorch, segmentation-models-pytorch | UNet with ResNet34 encoder |
| **Inference** | TorchScript | Optimized production model |
| **Classical CV** | OpenCV, NumPy | 5 thresholding methods + CLAHE preprocessing |
| **Image Processing** | OpenCV, Pillow | Overlay, heatmap, encoding |

---

## 📦 Installation

### Prerequisites

- **Python 3.10+** · **Node.js 18+** · **GPU** (optional, falls back to CPU)

### 1. Clone

```bash
git clone https://github.com/your-username/ATLAS.git
cd ATLAS
```

### 2. Backend

```bash
cd backend
python -m venv venv
.\venv\Scripts\activate          # Windows
# source venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
```

### 3. Model Files

Place in `backend/models/`:

```
backend/models/
├── model.pt              # TorchScript model (~93 MB)
└── model_config.json     # Model configuration
```

> Don't have a model? Train your own — see [Training](#-training) below.

### 4. Frontend

```bash
cd frontend
npm install
```

---

## 🚀 Quick Start

**Terminal 1 — Backend:**

```bash
cd backend
.\venv\Scripts\activate
uvicorn main:app --reload --port 8000
```

**Terminal 2 — Frontend:**

```bash
cd frontend
npm run dev
```

Open **http://localhost:3000** in your browser.

---

## 🏋️ Training

Train your own UNet model using `files/production_model_training.py`. Works on **Google Colab** (just `%run`) or **locally** via CLI.

### Dataset Structure

```
<data_dir>/
├── train/
│   ├── img/       ← training images (.png/.jpg)
│   └── label/     ← corresponding label masks
└── val/
    ├── img/       ← validation images
    └── label/     ← corresponding label masks
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `./datasets` | Root dataset directory (contains train/ and val/) |
| `--output_dir` | `./output` | Where model checkpoints and exports are saved |
| `--zip_path` | `None` | Optional: path to a dataset .zip to auto-extract |
| `--epochs` | `50` | Number of training epochs |
| `--batch_size` | `16` | Batch size (reduce if OOM) |
| `--lr` | `1e-4` | Learning rate |
| `--encoder` | `resnet34` | Backbone: `resnet34`, `efficientnet-b0`, `mobilenet_v2` |
| `--img_size` | `256` | Input image size (height = width) |

### Examples

```bash
# Default (expects ./datasets/ with train/ and val/)
python files/production_model_training.py

# Custom paths
python files/production_model_training.py \
    --data_dir /path/to/my_data \
    --output_dir ./my_model \
    --epochs 30 \
    --batch_size 8

# Extract from zip first
python files/production_model_training.py \
    --zip_path /path/to/dataset.zip \
    --data_dir ./extracted

# Google Colab
%run files/production_model_training.py
```

After training, copy the exported files to the backend:
```bash
cp output/exports/model.pt backend/models/
cp output/exports/model_config.json backend/models/
```

---

## 🔮 Inference (CLI)

Run predictions from the command line using `files/model_inference.py`.

### CLI Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--model` | ✅ | — | Path to model file (`.pt` or `.pth`) |
| `--config` | ✅ | — | Path to `model_config.json` |
| `--image` | — | — | Path to a single image |
| `--image_dir` | — | — | Path to a directory of images |
| `--threshold` | — | `0.5` | Prediction threshold |
| `--output_dir` | — | `./inference_output` | Where to save results |
| `--device` | — | `cuda` | `cuda` or `cpu` |

### Examples

```bash
# Single image
python files/model_inference.py \
    --model backend/models/model.pt \
    --config backend/models/model_config.json \
    --image road_photo.jpg

# Batch directory
python files/model_inference.py \
    --model backend/models/model.pt \
    --config backend/models/model_config.json \
    --image_dir ./test_images/ \
    --output_dir ./results/

# Custom threshold on CPU
python files/model_inference.py \
    --model backend/models/model.pt \
    --config backend/models/model_config.json \
    --image test.jpg \
    --threshold 0.6 \
    --device cpu
```

Outputs for each image: `<name>_mask.png` + `<name>_overlay.png` saved to `--output_dir`.

---

## 🖥️ Pages & Features

### 1. 🏠 Home (`/`)
Hero section with project overview, stats bar (85%+ IoU, ~15ms inference), and feature cards.

### 2. 🔍 Segment (`/segment`)
The main interactive page with everything:
- **Upload** — Drag-and-drop or browse
- **4-Tab Viewer** — Overlay, Mask, Heatmap, Original
- **Metrics** — Inference time, road coverage %, confidence
- **Download** — Individual PNGs or full ZIP
- **Pipeline Visualizer** — See all 6 preprocessing/inference steps
- **Threshold Grid** — Compare 6 thresholds (0.3–0.8) at once
- **Canvas Annotation** — Draw ground truth → compute IoU & Dice
- **Session History** — Gallery of past predictions

### 3. ⚔️ Compare (`/compare`)
Upload one image, see **UNet vs 5 classical methods** in a 6-card grid. Toggle between overlay and mask views. Each card shows method name, speed, and road coverage.

### 4. 📦 Batch (`/batch`)
Upload multiple images at once. Shows:
- Summary stats (avg inference time, avg road %, avg confidence)
- Per-image results with overlay thumbnails
- **CSV report download** with all metrics

### 5. ℹ️ About (`/about`)
Architecture diagram, method comparison table, model specs, and team info.

---

## 📡 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/model/info` | GET | Model configuration |
| `/predict` | POST | Single image → mask + overlay + heatmap + metrics |
| `/compare` | POST | UNet + 5 classical methods on one image |
| `/pipeline` | POST | 6-step pipeline visualization |
| `/threshold-grid` | POST | Results at 6 different thresholds |
| `/compute-metrics` | POST | IoU & Dice between two masks |
| `/predict/batch` | POST | Multi-image prediction |
| `/download` | POST | ZIP with mask, overlay, heatmap, metrics JSON |

Full interactive docs at **http://localhost:8000/docs**

---

## 🧠 Model Details

| Property | Value |
|----------|-------|
| Architecture | UNet |
| Encoder | ResNet34 (ImageNet pre-trained) |
| Input | 256 × 256 px (RGB) |
| Output | Binary mask (road / non-road) |
| Loss | BCE + Dice (combined) |
| Best Val IoU | 0.85+ |
| Inference | ~15ms (GPU) / ~150ms (CPU) |
| Export | TorchScript (.pt), ONNX (.onnx) |
| Size | ~93 MB |

---

## 🔬 Segmentation Methods

| Method | Avg IoU | Speed | Best For | Type |
|--------|---------|-------|----------|------|
| **UNet (ResNet34)** ⭐ | **0.85+** | **~15ms** | **All conditions** | Deep Learning |
| Adaptive Gaussian | 0.768 | 16ms | Shadows | Classical |
| Otsu | 0.756 | 12ms | Bright scenes | Classical |
| Adaptive Mean | 0.742 | 16ms | General | Classical |
| Sauvola | 0.721 | 45ms | Night scenes | Classical |
| Niblack | 0.698 | 43ms | Edge emphasis | Classical |

---

## 📊 Results

```
UNet (ours):      ████████████████████ IoU 0.85   ← Production model
Adaptive Gauss:   ███████████████▎     IoU 0.768
Otsu:             ███████████████      IoU 0.756
Adaptive Mean:    ██████████████▊      IoU 0.742
Sauvola:          ██████████████▍      IoU 0.721
Niblack:          █████████████▉       IoU 0.698
```

---

## 📁 Project Structure

```
ATLAS/
├── README.md
├── .gitignore
│
├── backend/                          # FastAPI Backend
│   ├── main.py                       # 7 API endpoints
│   ├── inference.py                  # UNet model wrapper
│   ├── classical_methods.py          # 5 thresholding methods + metrics
│   ├── requirements.txt
│   └── models/                       # Model files (gitignored)
│       ├── model.pt
│       └── model_config.json
│
├── frontend/                         # Next.js Frontend
│   └── app/
│       ├── layout.tsx                # Navbar + footer
│       ├── page.tsx                  # Home page
│       ├── globals.css               # Dark theme (1300+ lines)
│       ├── segment/page.tsx          # Segmentation + all interactive tools
│       ├── compare/page.tsx          # Method comparison
│       ├── batch/page.tsx            # Batch processing
│       └── about/page.tsx            # About page
│
└── files/                            # Training Scripts
    ├── production_model_training.py
    ├── model_inference.py
    └── MODEL_TRAINING_README.md
```

---

## 👥 Team

**ATLAS Development Team** — IIT Madras

| Member | Role |
|--------|------|
| **Jaswanth Prasanna V** | Full-Stack Development, Model Training |
| **Divya R** | Research, Classical Methods |
| **Haripriya K** | Research, Evaluation |

---

## 🙏 Acknowledgments

- **segmentation-models-pytorch** — UNet architecture
- **Cityscapes Dataset** — Road segmentation annotations
- **Otsu (1979)** — Threshold selection method
- **Sauvola et al. (2000)** — Adaptive document binarization

**Technologies:** PyTorch · FastAPI · Next.js · OpenCV · TorchScript

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 📚 Citation

```bibtex
@misc{atlas2026,
  title={ATLAS: Adaptive Thresholding with Language-Augmented Sensing},
  author={Jaswanth Prasanna V and Divya R and Haripriya K},
  year={2026},
  institution={IIT Madras},
  note={Full-stack AI road segmentation with UNet, classical methods, and interactive web UI}
}
```

---

<div align="center">

**🗺️ ATLAS — Mapping Roads with Intelligence**

Built with ❤️ at IIT Madras

</div>
