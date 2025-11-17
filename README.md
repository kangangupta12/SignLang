# Sign Language Recognition (CNN + LSTM)

Full-stack sign language recognition system that ingests raw videos, trains a CNN + BiLSTM sequence classifier, evaluates performance, and serves real-time predictions through a Streamlit web app (webcam + file upload). Everything runs locally, inside Docker, or on Streamlit Cloud with the same codebase.

---

## ✨ Highlights
- **Data pipeline** – Handles MP4 videos or pre-extracted frame folders, supports deterministic splits, on-the-fly augmentation, caching, and prefetching through `tf.data`.
- **Model** – MobileNetV2 (or custom lightweight CNN) wrapped in `TimeDistributed` + Bidirectional LSTM with dropout, batch norm, mixed-precision toggle, and SavedModel export.
- **Tooling** – Training CLI with checkpoints, TensorBoard, EarlyStopping, ReduceLROnPlateau; evaluation CLI with confusion matrix + per-class metrics; inference helpers for video/webcam.
- **Product experience** – Streamlit UI (model selector, webcam, upload, sample gallery), Docker image, synthetic data generator, benchmarking script, pytest coverage, and setup notebooks.

---

## 📁 Repository Layout
```
.
├── app/                    # Streamlit interface
├── artifacts/              # Label encoders, histories, eval assets
├── examples/               # Synthetic data generator
├── models/                 # Checkpoints / SavedModels
├── notebooks/              # EDA & mini-training walkthroughs
├── scripts/                # Helper bash scripts (train / streamlit / docker)
├── src/                    # Library code (config, data, model, train, eval, infer, utils, benchmark)
├── tests/                  # Pytest suites for data + model
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 📦 Setup
```bash
python -m venv .venv
. .venv/bin/activate        # Windows: .\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Configuration
All defaults live in `src/config.py` (overridable via `.env` or CLI flags):

| Key | Default | Description |
|-----|---------|-------------|
| `IMAGE_SIZE` | 224 | Input resolution (HxW) |
| `SEQ_LEN` | 30 | Frames per clip |
| `BATCH_SIZE` | 8 | Training batch size |
| `EPOCHS` | 30 | Max training epochs |
| `LEARNING_RATE` | 1e-4 | Adam lr |
| `CACHE_DATA` | true | Cache tf.data pipeline |
| `USE_MIXED_PRECISION` | false | Enable AMP on GPUs |
| `CNN_BACKBONE` | MobileNetV2 | Or `Custom` |
| `CNN_TRAINABLE` | false | Fine-tune backbone |

---

## 📚 Dataset Guidance
Recommended public sources:
- [WLASL](https://dxli94.github.io/WLASL/) – word-level ASL (~2k glosses)
- [LSA64](https://grfia.dlsi.ua.es/lsa64/) – 64 Argentinian signs
- [RWTH-PHOENIX-Weather 2014](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/) – weather broadcast signing

Organize either **videos** or **frames**:
```
dataset/
├── videos/
│   ├── hello/
│   │   ├── sample_001.mp4
│   │   └── ...
│   └── thanks/...
└── frames/
    ├── hello/sample_001/frame_000.jpg
    └── ...
```
Toggle modes via `--use_frames` or `USE_FRAMES=true`.

Need a quick sandbox? Generate synthetic clips (moving rectangles) that behave like short gestures:
```bash
python examples/generate_synthetic_sequence.py \
  --output_dir dataset/videos \
  --num_classes 3 \
  --samples_per_class 25
```

---

## 🚀 Training
```bash
python -m src.train \
  --dataset_dir dataset/videos \
  --epochs 30 \
  --batch_size 8 \
  --seq_len 30 \
  --checkpoint_dir models/checkpoints \
  --mixed_precision true         # optional
# add --use_frames for frame datasets
```
Artifacts:
- Checkpoints → `models/checkpoints/best.ckpt`
- Label encoder → `artifacts/label_encoder.json`
- History → `artifacts/history.json`
- TensorBoard logs → `artifacts/logs/<timestamp>`

View logs:
```bash
tensorboard --logdir artifacts/logs
```

---

## 📊 Evaluation
```bash
python -m src.eval \
  --dataset_dir dataset/videos \
  --checkpoint_path models/checkpoints/best.ckpt
```
Outputs:
- `artifacts/metrics_report.csv`
- `artifacts/confusion_matrix.png`

---

## 🔎 Inference & Utilities
```bash
# Single video
python -m src.infer --model_path models/checkpoints/best.ckpt --video_path path/to/video.mp4

# Webcam loop
python -m src.infer --model_path models/checkpoints/best.ckpt --webcam

# Benchmark FPS
python -m src.benchmark --model_path models/checkpoints/best.ckpt --seq_len 30 --num_runs 100
```

---

## 🌐 Streamlit App
```bash
streamlit run app/streamlit_app.py
# or
./scripts/run_streamlit.sh        # adds Docker option
```
Features:
- Select any checkpoint (local file picker or upload)
- Live webcam predictions via `streamlit-webrtc`
- Video upload with top‑3 probabilities and preview
- Demo gallery (drop sample MP4s into `examples/`)

When running outside the repo root, set `PYTHONPATH` accordingly so `src` imports resolve.

---

## 🐳 Docker
```bash
./scripts/build_docker.sh
docker run --rm -p 8501:8501 signlang:latest
```
Image boots directly into Streamlit (`app/streamlit_app.py`).

---

## ✅ Tests & Quality
```bash
pytest
```
Pytest covers:
- tf.data pipeline shape/label integrity
- Model output signatures & build logic
- Utility helpers (frame extraction, label encoding)

---

## 🔧 Fine-tuning & Performance Tips
- Unfreeze deeper MobileNetV2 blocks with a lower `learning_rate`.
- Increase `SEQ_LEN` or sampling FPS for longer gestures.
- Use `--mixed_precision` on Ampere+ GPUs to halve VRAM footprint.
- Export to SavedModel for serving:
  ```bash
  python -m src.train --export_saved_model models/saved_model
  ```
- Add class-balanced sampling or focal loss for skewed vocabularies.

Expected accuracy on modest vocabularies (30–80 glosses) with MobileNetV2+BiLSTM typically lands in the 80–90 % range; larger vocabularies benefit from additional data augmentation and longer training schedules.

---

## 🧠 Hardware Notes
- **CPU-only**: fine for experimentation & the synthetic dataset; training real corpora will be slow.
- **GPU (≥8 GB VRAM)**: recommended for production-sized vocabularies; enable mixed precision.
- **Webcam inference**: runs on CPU by default; GPU accelerates the per-frame CNN pass.

---

## 📄 License
MIT – feel free to fork, improve, and deploy. Pull requests welcome!
