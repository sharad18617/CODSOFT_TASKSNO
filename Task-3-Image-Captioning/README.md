# AI Image Captioning System
### CodSoft Artificial Intelligence Internship — Task 3

[![Live Demo](https://img.shields.io/badge/Streamlit%20Cloud-Live%20Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://codsoft-ai.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/Salesforce/blip-image-captioning-base)

> 🌐 **Live Web Application:** [https://codsoft-ai.streamlit.app](https://codsoft-ai.streamlit.app)

An end-to-end deep learning application that combines **Computer Vision (CV)** and **Natural Language Processing (NLP)** to generate descriptive, natural-language captions for images. The project uses a pre-trained **Salesforce BLIP (Bootstrapping Language-Image Pre-training)** Vision-Language Transformer model via Hugging Face Transformers, PyTorch, and an interactive Streamlit web dashboard.

---

## 📌 Project Overview

Image captioning is a multimodal AI task that requires an algorithm to recognize objects, understand spatial relationships, and construct meaningful sentences describing a scene.

This project implements a complete local captioning pipeline:
1. Accepts an uploaded image (JPG, JPEG, PNG, WEBP).
2. Extracts visual feature representations using a **Vision Transformer (ViT)** encoder.
3. Generates natural-language sentences using a **Transformer Language Decoder** with cross-attention.
4. Provides real-time metrics including **inference latency** and the active **compute device** (Apple Silicon MPS / CUDA / CPU).

---

## 🚀 Key Features

- **Pretrained Vision-Language Model:** Uses `Salesforce/blip-image-captioning-base` from Hugging Face for caption generation without training from scratch.
- **100% Free & Local Execution:** Runs completely offline after the initial model download; requires no paid APIs or API keys.
- **Hardware Acceleration with Fallback:** Automatically detects and utilizes **Apple Silicon GPU (MPS - Metal Performance Shaders)** or **NVIDIA CUDA GPU**, with seamless fallback to **CPU**.
- **Interactive Streamlit Web Dashboard:** Modern, uncluttered interface designed for quick image uploading and caption inspection.
- **Dual Captioning Modes:**
  - *Standard Mode (Unconditional):* Generates a complete description of the scene from scratch.
  - *Guided / Prefix Mode (Conditional):* Directs caption generation with a starting prompt (e.g., `"A close-up photo of"`).
- **Configurable Beam Search:** Adjustable beam width slider to balance inference speed versus caption quality.
- **In-Memory Model Caching:** Uses Streamlit's `@st.cache_resource` decorator to load the ~990 MB model into RAM once across sessions.
- **Copy Caption Helper:** Quick-copy text field to easily copy generated captions to clipboard.
- **CLI & Scripting Support:** Includes a standalone command-line interface in `caption_generator.py` for terminal usage.

---

## 🧠 How It Works / System Architecture

The BLIP architecture bridges the gap between vision and language through a unified encoder-decoder framework:

```
[Input Image (JPG/PNG/WEBP)]
             │
             ▼
   ┌───────────────────┐
   │   BlipProcessor   │ ──► Resizes image to 384×384 and normalizes pixel values
   └───────────────────┘
             │
             ▼
┌─────────────────────────┐
│ Vision Transformer(ViT) │ ──► Splits image into 16×16 patches & extracts visual feature embeddings
└─────────────────────────┘
             │
             ▼
┌─────────────────────────┐
│ Text Transformer Decoder│ ──► Cross-attention combines visual features with text tokens
└─────────────────────────┘
             │
             ▼
   ┌───────────────────┐
   │    Beam Search    │ ──► Evaluates multiple sentence candidate paths
   └───────────────────┘
             │
             ▼
[Natural Language Caption Output]
```

### Technical Workflow:
1. **Preprocessing:** The `BlipProcessor` resizes the input image to $384 \times 384$ pixels and normalizes color channels according to ImageNet statistics.
2. **Visual Feature Extraction:** The **Vision Transformer (ViT)** splits the image into non-overlapping $16 \times 16$ pixel patches and projects them into dense visual embedding vectors.
3. **Cross-Attention Decoding:** The language decoder processes text tokens autoregressively while attending to the image embeddings through cross-attention layers.
4. **Token Generation (Beam Search):** Beam search evaluates top candidate word sequences at each generation step to produce coherent, grammatically accurate sentences.
5. **Post-Processing:** The generated token IDs are decoded back into a human-readable string and displayed in the UI.

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.9+ | Primary programming language |
| **Deep Learning** | PyTorch (`torch`, `torchvision`) | Tensor operations & hardware acceleration (MPS/CUDA/CPU) |
| **Model & Tokenizer** | Hugging Face `transformers` | Pretrained `Salesforce/blip-image-captioning-base` architecture |
| **Image Processing** | Pillow (`PIL`) | Image loading, format validation, and RGB conversion |
| **User Interface** | Streamlit | Interactive web dashboard and reactive user controls |

---

## 📂 Project Structure

```
Task-3-Image-Captioning/
├── app.py                   # Streamlit web application with UI layout & session caching
├── caption_generator.py     # Core ImageCaptioner class & standalone CLI tool
├── utils.py                 # Hardware detection (MPS/CUDA/CPU) and image validation helpers
├── requirements.txt         # Pinned Python package dependencies
├── .gitignore               # Excludes virtual environment, model caches, and logs
└── README.md                # Comprehensive project documentation
```

### Module Breakdown:
- **`caption_generator.py`:** Contains the `ImageCaptioner` class that manages model loading, tensor preparation, beam search generation, and standalone CLI arguments (`--image`, `--prefix`, `--beams`).
- **`utils.py`:** Provides device selection logic (`get_device`, `get_device_name`) and image validation (`validate_and_load_image`) ensuring images are converted to 3-channel RGB.
- **`app.py`:** Renders the responsive Streamlit dashboard, sidebar controls, progress spinners, and result metrics.

---

## 💻 Installation & Setup

### 1. Clone or Navigate to the Project Directory
```bash
cd Task-3-Image-Captioning
```

### 2. Create and Activate a Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate on macOS / Linux:
source venv/bin/activate

# Activate on Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on Initial Run:** On the first execution, Hugging Face will download the pre-trained BLIP model weights (~990 MB). The weights are cached locally in `~/.cache/huggingface/` and loaded directly on subsequent runs.

---

## 🚀 How to Run the Application

### Option A: Launch the Streamlit Web Dashboard (Recommended)
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

1. Upload an image (`.jpg`, `.jpeg`, `.png`, or `.webp`).
2. Adjust sidebar settings if desired (Standard vs. Guided Mode, Beam Width).
3. Click **Generate Caption** to view the description, latency, and active compute device.

---

### Option B: Run via Command-Line Interface (CLI)
You can generate captions directly from the terminal without starting the web server:

```bash
# Standard unconditional captioning
python caption_generator.py --image path/to/sample.jpg

# Guided conditional captioning with a prefix prompt
python caption_generator.py --image path/to/sample.jpg --prefix "A photo of"

# Custom beam search width
python caption_generator.py --image path/to/sample.jpg --beams 5
```

---

## 📊 Example Outputs

| Input Image Type | Captioning Mode | Generated Caption |
| :--- | :--- | :--- |
| **Park Scene** | Standard | *"A golden retriever running across a green lawn with a tennis ball in its mouth"* |
| **City Skyline** | Standard | *"A wide view of modern skyscrapers lit up at dusk reflected in the river below"* |
| **Food / Dish** | Guided (`"A close-up photo of"`) | *"A close-up photo of a bowl of fresh ramen topped with sliced pork and green onions"* |

---

## ⚡ Performance & Hardware Note

- **Apple Silicon (M-series Macs):** Automatically detected as `Apple Silicon GPU (MPS Accelerated)`. Inferences typically complete within **0.5 – 1.5 seconds**.
- **NVIDIA GPUs:** Automatically detected with CUDA acceleration enabled.
- **CPU Fallback:** Runs smoothly on standard CPU architectures without code modifications.
- **Memory Management:** The Streamlit application utilizes `@st.cache_resource` so the model remains in RAM/VRAM across user interactions, avoiding redundant reloading overhead.

---

## 🎯 CodSoft Task Requirements Alignment

This project directly fulfills the requirements for **CodSoft AI Internship — Task 3 (Image Captioning)**:

| Task Specification | Project Implementation |
| :--- | :--- |
| **Combine Computer Vision and NLP** | Combines a Vision Transformer (ViT) patch encoder with a text Transformer decoder. |
| **Use Pretrained Feature Extractor** | Uses Salesforce BLIP Base via Hugging Face Transformers. |
| **Transformer/RNN-based Captioning** | Utilizes a Transformer decoder with cross-attention and beam search decoding. |
| **Working User Interface** | Provides an interactive, modern Streamlit web dashboard. |
| **Local / Accessible Execution** | 100% open-source, runs locally without paid API subscriptions or keys. |

---

## 🔮 Future Improvements

- [ ] **Fine-Tuning:** Fine-tune model weights on domain-specific datasets (e.g., medical radiology, architectural imagery).
- [ ] **Batch Processing:** Allow users to upload folders of images and export captions to CSV/JSON.
- [ ] **Accessibility (TTS):** Integrate a Text-to-Speech (TTS) audio readout module for visually impaired users.
- [ ] **Multilingual Captions:** Integrate translation models (e.g., MarianMT) to output captions in multiple languages.

---

## 👤 Author

**Sharad Panchal**
- **Internship:** CodSoft Artificial Intelligence Internship
- **Task:** Task 3 — AI Image Captioning System
- **Repository:** [CODSOFT_TASKSNO](https://github.com/sharadpanchal/CODSOFT_TASKSNO)
