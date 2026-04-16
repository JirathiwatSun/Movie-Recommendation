# 🎬 CineMatch 2.0: AI-Powered Movie Discovery Platform

> **ITCS227: Introduction to Data Science & Artificial Intelligence**
> A production-grade streaming-style application delivering weighted statistical rankings, NLP semantic "vibe" search, and interactive 3D cinematic relationship mapping.

CineMatch 2.0 is an advanced content-based recommendation engine built with **Streamlit** and **Python**. It features a state-of-the-art AI core that transcends simple keyword matching to understand the "soul" of cinema through high-dimensional embeddings and interactive network analysis.

---

## 📋 Prerequisites

Before starting, ensure you have the following installed:
- **Python 3.9+** (Tested on Python 3.11)
- **Git & Git LFS** (Required for pre-trained models)
  - Windows: [Download Git LFS](https://git-lfs.github.com/)
- **Kaggle Account** (Optional, but `kagglehub` manages the dataset automatically)

---

## 📦 Installation & Quick Start

Follow these steps to set up the environment and launch the CineMatch platform:

### 1. Clone the repository
```bash
git clone https://github.com/9gatsu28nichi/Movie-Recommendation.git
cd Movie-Recommendation
```

### 2. Pull Pre-trained AI Models (Important)
CineMatch uses large model files (~1.2GB). Ensure they are downloaded correctly via Git LFS:
```bash
git lfs install
git lfs pull
```

### 3. Create and Activate Virtual Environment
```bash
python -m venv .venv

# On Windows:
.\.venv\Scripts\activate

# On Unix or MacOS:
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Initialize the Dataset & Health Check
Run the automated installation script to download the database and verify model integrity:
```bash
python install_data.py
```
> [!NOTE]
> This script downloads the movie dataset into `archive/` and performs a "hydration" check on your AI models to ensure they aren't just empty pointers.

### 6. Launch the Platform
```bash
streamlit run app.py
```

---

## 🧠 Key Features (Oracle Edition)

- **🔮 CineMatch Oracle**: Intent-based discovery that understands "Vibe" (Visual Style, Emotional Impact, Narrative Complexity).
- **⚡ Mood Fusion Engine**: Identify films sitting at the intersection of multiple moods.
- **💎 Hidden Gem Radar**: Surfaces high-quality independent films (>8.0 IMDB) often buried by blockbusters.
- **🕸️ 3D Cinematic Explorer**: Interactive relationship mapping between 436,000+ movies via Plotly.

---

## 🛠️ Tech Stack

- **UI/UX**: Streamlit, Vanilla CSS, Plotly.
- **Core AI**: Sentence Transformers (all-MiniLM-L6-v2), PyTorch, Scikit-Learn.
- **Data**: Pandas, Numpy, NLTK (WordNet), Kaggle Hub.
- **Visualization**: Seaborn, Matplotlib.

---

## 🔧 Troubleshooting

### 1. AI Engine Initializing on CPU
If the app states it's loading on the CPU instead of your GPU (NVIDIA only), run:
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. Redundant AI Retraining
If every search triggers a "Scanning fingerprints" progress bar (takes ~5-10 mins), your `models/` folder contains Git LFS pointers instead of binaries.
**Fix:** Run `git lfs pull` and then `python install_data.py`.

---

## 🔬 Technical Report
For a deep dive into the math and algorithms (IMDB Weighted Ratings, Semantic Soup Generation, etc.), view the **[MovieRecommendation.ipynb](MovieRecommendation.ipynb)**.

---
*Created by Jirathiwat Sun for ITCS227.*
