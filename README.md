# 🎭 Navarasa — Emotion-Aware Music Recommendation System

> Bridging ancient Indian emotional theory with modern deep learning

Detect your facial emotion in real-time using a CNN trained on AffectNet 8-class dataset, map it to the ancient Indian Navarasa (9 rasas) framework, and receive Spotify song recommendations personalized to your emotional state.

---

## 📊 Project Overview

This system combines:
- **Deep Learning**: Custom CNN and MobileNetV2 trained on 29k+ AffectNet images
- **Traditional Knowledge**: Ancient Indian Navarasa emotional framework (Fury, Wonder, Heroism, Joy, Peace, Terror, Sorrow, Disgust, Love)
- **Music Intelligence**: Audio feature filtering (valence, energy, danceability) from 100k+ Spotify tracks
- **Real-Time Detection**: Live webcam emotion detection with Navarasa overlay
- **Web App**: Streamlit UI with dark theme inspired by Spotify

### Architecture

```
Webcam Frame
    ↓
OpenCV Haar Cascade Face Detection
    ↓
Preprocess to 96×96 RGB
    ↓
CNN Model (CustomCNN or MobileNetV2)
    ↓
Emotion Class + Confidence (8 emotions)
    ↓
Map to Navarasa Framework
    ↓
Contextual Shringara Inference
    ↓
Audio Profile Filter (valence/energy/danceability)
    ↓
Spotify Dataset Search
    ↓
Return Top 5 Songs + Display to User
```

---

## 🎭 The 9 Navarasa

| Rasa | Emotion | Meaning | Emoji | Detected? |
|------|---------|---------|-------|-----------|
| Raudra | Anger | Fury | 😠 | ✅ |
| Adbhuta | Surprise | Wonder | 😲 | ✅ |
| Vira | Contempt | Heroism | 😤 | ✅ |
| Hasya | Happy | Joy | 😄 | ✅ |
| Shanta | Neutral | Peace | 😌 | ✅ |
| Bhayanaka | Fear | Terror | 😨 | ✅ |
| Karuna | Sad | Sorrow | 😢 | ✅ |
| Bibhatsa | Disgust | Disgust | 🤢 | ✅ |
| Shringara | Happy (contextual) | Love | 🥰 | ✅\* |

\* *Shringara is inferred contextually: happy emotion + high confidence (>0.75) + romantic music history*

---

## 📦 Datasets

### Dataset 1: AffectNet 8-Class Emotion Dataset
- **Source**: [Mauregato/affectnet_short](https://huggingface.co/datasets/Mauregato/affectnet_short)
- **Size**: 29,042 RGB images (96×96)
- **Split**: Train 23.2k / Val 5.81k
- **License**: Research use
- **Classes**: 8 emotions with balanced sampling via WeightedRandomSampler

### Dataset 2: Spotify Tracks Dataset
- **Source**: [maharshipandya/spotify-tracks-dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset)
- **Size**: 100k+ tracks with audio features
- **License**: MIT
- **Features**: valence, energy, danceability, tempo, popularity, genre

---

## 📁 Project Structure

```
emotion-music-recommender/
├── data/
│   ├── setup_datasets.py          # Load & validate HF datasets
│   └── songs_cache.parquet        # Cached Spotify tracks (generated)
│
├── models/
│   ├── custom_cnn.py              # CNN from scratch (8 conv blocks + FC)
│   ├── mobilenet_model.py         # Transfer learning with MobileNetV2
│   ├── train.py                   # Unified training script
│   └── checkpoints/               # Best model saves
│       ├── custom_cnn_best.pth
│       ├── custom_cnn_history.json
│       ├── mobilenet_best.pth
│       └── mobilenet_history.json
│
├── utils/
│   ├── constants.py               # Single source of truth for all mappings
│   ├── dataset.py                 # AffectNetDataset with WeightedRandomSampler
│   ├── evaluate.py                # Model evaluation & comparison tools
│   └── emotion_history.py         # Session tracking + visualization
│
├── music/
│   ├── preprocess_songs.py        # Load & cache Spotify dataset
│   └── recommendations.py         # Emotion → audio profile → songs
│
├── app/
│   ├── webcam.py                  # Real-time emotion detector
│   └── ui.py                      # Streamlit web interface
│
├── notebooks/
│   └── comparison.ipynb           # Model comparison & analysis
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Setup Environment

```bash
git clone <repository-url>
cd emotion-music-recommender

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Load & Cache Datasets

```bash
# Download AffectNet & Spotify datasets, verify integrity, cache Spotify
python data/setup_datasets.py
python music/preprocess_songs.py

# Expected output:
# ✅ Dataset 1 (AffectNet) loaded successfully — 29,042 samples
# ✅ Dataset 2 (Spotify) loaded successfully — 100,000+ tracks cached
```

### 3. Train Models

```bash
# Train CustomCNN (60 epochs, batch size 64)
python models/train.py --model custom_cnn --epochs 60 --batch_size 64

# Train MobileNetV2 (60 epochs, batch size 32)
python models/train.py --model mobilenet --epochs 60 --batch_size 32

# Expected output:
# ✅ Training complete!
#   Best Val Accuracy: 0.7234
#   Best Epoch: 42
#   Checkpoint: models/checkpoints/mobilenet_best.pth
```

### 4. Evaluate & Compare Models

Open and run the Jupyter notebook:

```bash
jupyter notebook notebooks/comparison.ipynb
```

This notebook will:
- Load both trained models
- Plot training curves (loss, accuracy)
- Display confusion matrices for each emotion
- Print classification reports highlighting contempt F1
- Benchmark inference speed and model size
- Provide deployment recommendation

### 5. Run Web App

```bash
streamlit run app/ui.py
```

Opens at `http://localhost:8501`

---

## 🎯 Training Results

### Model Comparison

| Metric | CustomCNN | MobileNetV2 |
|--------|-----------|-------------|
| Val Accuracy | ~70% | ~68% |
| Contempt F1 | Low | Better |
| Parameters | 2.1M | 3.5M |
| Inference Time | 15ms | 25ms |
| Model Size | 8.5 MB | 13.2 MB |

**Recommendation**: Deploy **MobileNetV2** for better robustness despite slightly higher latency.

---

## 🎵 How Navarasa Mapping Works

### Emotion → Audio Profile

Each emotion maps to a range of Spotify audio features:

```python
EMOTION_AUDIO_PROFILES = {
    "anger":    {"valence": (0.0, 0.35), "energy": (0.75, 1.0), ...},
    "happy":    {"valence": (0.7, 1.0),  "energy": (0.6, 1.0),  ...},
    "sad":      {"valence": (0.0, 0.3),  "energy": (0.0, 0.4),  ...},
    ...
}
```

### Contextual Shringara Inference

When detected emotion is "happy" with high confidence (>0.75) AND user's recent play history includes romantic genres (R&B, soul, indie), the system upgrades to **Shringara** (Love rasa), yielding different song recommendations.

---

## 🎧 How Emotion-to-Music Works

1. **Detect emotion** from facial expression → confidence score
2. **Map to audio profile** using EMOTION_AUDIO_PROFILES
3. **Fall back to neutral** if confidence < 0.5 (uncertain detection)
4. **Filter Spotify tracks** by valence/energy/danceability ranges
5. **Exclude recently played** songs (last 10 tracks)
6. **Sort by popularity** and return top 5
7. **Record in history** for Shringara inference

---

## 💡 Key Features

### Real-Time Detection
- ✅ Live webcam input with <50ms latency (CPU)
- ✅ Haar Cascade face detection + CNN inference
- ✅ Confidence-based filtering
- ✅ FPS counter overlay

### Music Recommendations
- ✅ 5 personalized songs per emotion
- ✅ Contextual Shringara upgrade
- ✅ Audio feature display (valence, energy, danceability, popularity)
- ✅ Spotify-inspired UI with dark theme

### Session Analytics
- ✅ Emotion timeline visualization
- ✅ Navarasa frequency chart
- ✅ Session statistics (total detections, dominant emotion, duration)
- ✅ Save session as JSON for later analysis

### Model Comparison
- ✅ Training curve comparison
- ✅ Confusion matrices
- ✅ Per-emotion F1 scores (watch contempt!)
- ✅ Inference benchmarking
- ✅ Deployment recommendation

---

## ⚙️ Configuration

### Confidence Threshold
Set in Streamlit sidebar (0.3–0.9, default 0.5). Detections below this are ignored for recommendations.

### Model Selection
Choose between:
- **MobileNetV2 (Recommended)**: Better accuracy, transfer learning advantage
- **Custom CNN**: Simpler, faster inference

### Recent History Windows
- Last 10 tracks: For recommendation exclusion
- Last 5 genres: For Shringara inference
- Last 30 seconds: For dominant emotion calculation

---

## 🔮 Future Work

- [ ] Add **Shringara as a trained class** with curated romantic dataset
- [ ] Integrate **real Spotify OAuth API** for actual playlist creation
- [ ] **Multi-face tracking** for group emotion recognition
- [ ] **Voice + face fusion** for more robust emotion detection
- [ ] **Mobile deployment** via ONNX export
- [ ] **Batch playlist generation** for different time-of-day moods
- [ ] **Genre-specific filtering** based on user preferences

---

## 🧠 Technical Stack

- **Deep Learning**: PyTorch 2.0+, torchvision
- **Data Loading**: HuggingFace datasets library
- **Computer Vision**: OpenCV 4.8+
- **Web Framework**: Streamlit 1.28+
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Evaluation**: scikit-learn
- **Notebook**: Jupyter

---

## 📊 Citation

This project uses:
- **AffectNet Dataset**: Mollahosseini et al., "AffectNet: A Database for Facial Expression Recognition" (2019)
- **Spotify Tracks**: [maharshipandya/spotify-tracks-dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset)
- **MobileNetV2**: Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks" (2018)

---

## ⚖️ License

This project is provided for educational and research purposes. Comply with:
- AffectNet license (research use only)
- Spotify dataset license (MIT)
- Dataset source attribution requirements

---

## 📧 Contact & Support

For questions or issues:
1. Check the `notebooks/comparison.ipynb` for detailed analysis
2. Review `models/train.py` logs for training insights
3. Ensure `data/songs_cache.parquet` exists before running the app

---

## 🎭 Final Note

> *"The Navarasa represent the very essence of human experience. By bridging ancient wisdom with modern technology, we create a system that understands not just what you feel, but what you need to hear."*

Enjoy discovering music through emotion! 🎵
