# Speech Recognition and Enhancement for Hearing Aids

A hybrid CNN-BiLSTM deep learning system for real-time speech enhancement in hearing aid applications, combining traditional signal processing with advanced machine learning techniques.

## Overview

This project addresses communication challenges faced by hearing aid users by implementing an intelligent speech enhancement pipeline. The system preprocesses audio signals using MATLAB and applies a CNN-LSTM architecture in Python to significantly improve speech intelligibility in noisy environments.

## Key Features

- **Hybrid Architecture**: Combines CNN for spatial feature extraction and Bi-LSTM for temporal dependency modeling
- **MATLAB Preprocessing**: Advanced signal processing including filtering, spectral subtraction, and noise reduction
- **Binaural Processing**: Dual-channel architecture preserving spatial audio cues
- **Multiple Hearing Loss Profiles**: Trained on 7 different hearing loss patterns (mild, moderate, severe, high-frequency, etc.)
- **Real-time Performance**: ~100,000 parameters enabling efficient edge deployment
- **Robust Enhancement**: 2.3% average improvement in MBSTOI intelligibility scores

## System Architecture
```
Audio Input → MATLAB Preprocessing → CNN-BiLSTM Model → Enhanced Output
                    ↓                          ↓
            [Noise Reduction]          [Speech Separation]
            [Normalization]            [Feature Enhancement]
            [Spectral Subtraction]     [Channel Fusion]
```

## Technologies Used

- **MATLAB**: Signal preprocessing and feature extraction
- **Python**: Deep learning model implementation
- **TensorFlow/Keras**: Neural network framework
- **Jupyter Notebook**: Model development environment

##  Model Performance

- **Validation Loss**: 0.2672
- **Mean Absolute Error (MAE)**: 0.2236
- **Average MBSTOI Improvement**: +0.0232 (2.3%)
- **Success Rate**: 100% improvement across 1000 test samples
- **Training**: 159 epochs with early stopping

### Performance by Hearing Loss Type
- High-frequency loss: +0.0199 improvement
- Moderate loss: +0.0253 improvement
- Clean conditions: +0.0407 improvement
- Noisy conditions: +0.0271 improvement

## Getting Started

### Prerequisites
```bash
MATLAB R2020a or later
Python 3.8+
TensorFlow 2.x
NumPy
SciPy
Librosa
```

### Usage

1. **Preprocess Audio (MATLAB)**
```matlab
% Run preprocessing script
run('preprocessing/audio_preprocessing.m')
```

2. **Train Model (Python)**
```python
# Train the CNN-BiLSTM model
python train_model.py --epochs 200 --batch_size 32
```

3. **Enhance Audio**
```python
# Enhance speech from audio file
python enhance_audio.py --input noisy_audio.wav --output enhanced_audio.wav
```

## 📁 Project Structure
```
├── preprocessing/          # MATLAB preprocessing scripts
├── models/                 # CNN-BiLSTM architecture
├── data/                   # Dataset (not included)
├── notebooks/              # Jupyter notebooks
├── utils/                  # Helper functions
├── results/                # Output and evaluation metrics
└── report/                 # Project documentation
```

## 📈 Results

### Spectral Comparison
The model successfully reduces background noise while preserving speech clarity:
- Cleaner spectrograms with focused energy patterns
- Reduced noise spread across frequency bands
- Enhanced speech-dominant frequencies

### Audio Quality
- Noticeably clearer speech in noisy environments
- Improved intelligibility for various hearing loss profiles
- Consistent performance across different noise types

## 🔬 Technical Details

### Network Architecture
- **Input**: Dual-channel (left/right) with 5-dimensional feature vectors
- **Conv1D Layer**: 32 filters for feature extraction
- **BiLSTM Layer**: 128 units capturing temporal dependencies
- **Dense Layers**: 128 → 64 → 5 units per channel
- **Total Parameters**: 100,234

### Preprocessing Pipeline
- Resampling to 16 kHz
- High-pass filtering for low-frequency noise removal
- Pre-emphasis for high-frequency enhancement
- 40ms frames with 50% overlap
- Hamming windowing

## 🎯 Future Work

- [ ] Model compression for edge deployment
- [ ] Real-world environment testing
- [ ] User-specific adaptation mechanisms
- [ ] Smartphone-based interactive interface
- [ ] Performance optimization for reverberant conditions

## 👥 Authors

- **Sathwik Chowdary Merla** - merla.s@northeastern.edu
- **Rucha Bhandari** - bhandari.ru@northeastern.edu
- **Dharm Mehta** - mehta.dhar@northeastern.edu

*Northeastern University, Boston, MA*



---
