# Sign Language Recognition (CNN + LSTM)

Production-ready pipeline for dynamic sign language recognition using a hybrid CNN + LSTM built with TensorFlow/Keras. Includes dataset preparation utilities, training/evaluation scripts, Streamlit deployment (webcam + file upload), Docker support, benchmarking, tests, and reproducible notebooks.

## Features
- Modular 	f.data pipeline for videos or pre-extracted frames with caching/prefetch.
- Augmentation: random crop, flip, brightness jitter, small rotations.
- Configurable MobileNetV2 + Bidirectional LSTM with mixed-precision toggle.
- Training loop with checkpoints, TensorBoard, ReduceLROnPlateau, early stopping.
- Evaluation script outputs per-class metrics CSV + confusion matrix PNG.
- Streamlit app with webcam, upload, model selector, and optional sample demos.
- Synthetic data generator to bootstrap experiments without a full dataset.
- Dockerfile, helper scripts, pytest unit tests, benchmarking utility.

## File Structure
`
.
â”œâ”€â”€ src/
â”œâ”€â”€ app/
â”œâ”€â”€ notebooks/
â”œâ”€â”€ models/
â”œâ”€â”€ artifacts/
â”œâ”€â”€ tests/
â”œâ”€â”€ scripts/
â”œâ”€â”€ examples/
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ Dockerfile
â””â”€â”€ README.md
`

## Dataset Guidance
Suggested public datasets:
- **WLASL** (Word-Level American Sign Language): https://dxli94.github.io/WLASL/
- **LSA64** (Argentinian Sign Language): https://grfia.dlsi.ua.es/lsa64/
- **RWTH-PHOENIX-Weather 2014**: https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/

Organize videos or frames as:
`
dataset/
 â”œâ”€â”€ videos/
 â”‚    â”œâ”€â”€ hello/vid_0001.mp4
 â”‚    â””â”€â”€ ...
 â””â”€â”€ frames/
      â”œâ”€â”€ hello/vid_0001/frame_000.jpg
      â””â”€â”€ ...
`
Choose mode via --use_frames flag.

## Dependencies
See equirements.txt.
`
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
`

## Configuration
Defaults in src/config.py (override via .env or CLI):
`
IMAGE_SIZE=224
SEQ_LEN=30
BATCH_SIZE=8
EPOCHS=30
LEARNING_RATE=1e-4
CACHE_DATA=true
USE_MIXED_PRECISION=false
CNN_BACKBONE=MobileNetV2
CNN_TRAINABLE=false
`

## Training
`
python -m src.train \
  --dataset_dir dataset/videos \
  --use_frames false \
  --epochs 30 \
  --batch_size 8 \
  --seq_len 30 \
  --checkpoint_dir models/checkpoints \
  --mixed_precision true
`
Artifacts in models/checkpoints/, rtifacts/history.json, rtifacts/label_encoder.json, and TensorBoard logs under rtifacts/logs/.

### TensorBoard
`
tensorboard --logdir artifacts/logs
`

## Evaluation
`
python -m src.eval \
  --dataset_dir dataset/videos \
  --use_frames false \
  --seq_len 30 \
  --checkpoint_path models/checkpoints/best.ckpt
`
Outputs rtifacts/metrics_report.csv and rtifacts/confusion_matrix.png.

## Inference
`
python -m src.infer --model_path models/checkpoints/best.ckpt --video_path sample.mp4
python -m src.infer --model_path models/checkpoints/best.ckpt --webcam
`

## Streamlit App
`
streamlit run app/streamlit_app.py
# or
./scripts/run_streamlit.sh
`
Features: model selection (local/upload), webcam predictions via streamlit-webrtc, video upload with top-3 probabilities, optional demo thumbnails from examples/.

## Docker
`
./scripts/build_docker.sh
./scripts/run_streamlit.sh --docker
`

## Synthetic Data
`
python examples/generate_synthetic_sequence.py \
  --output_dir dataset/videos \
  --num_classes 2 \
  --samples_per_class 5
`

## Benchmarking
`
python -m src.benchmark --model_path models/checkpoints/best.ckpt --seq_len 30 --num_runs 100
`

## Tests
`
pytest
`

## Model Performance & Fine-tuning
Expect 80â€“90% accuracy on small vocabularies with MobileNetV2 + BiLSTM. Improve by increasing sequence length, unfreezing CNN layers, class-balanced sampling, or mixed-precision training. Export SavedModel with:
`
python -m src.train --export_saved_model models/saved_model
`

## Hardware Notes
- CPU works for demos but training is slow.
- GPU with â‰¥8 GB VRAM recommended; mixed precision needs CUDA CC â‰¥7.0.
- Webcam inference runs on CPU; GPU accelerates per-frame CNN.

## License
MIT License.
