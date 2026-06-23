# Music Genre Classifier: Messy Mashup 🎵

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-F9AB00.svg)](https://huggingface.co/)
[![Score](https://img.shields.io/badge/Private%20Score-0.9609-brightgreen.svg)]()

An end-to-end deep learning pipeline designed to accurately classify musical genres from severely corrupted 10-second audio clips. This project tackles the complex Music Information Retrieval (MIR) challenge of identifying genres despite overlapping multi-stem instrument bleed and extreme environmental noise.

---

## 🛑 The Challenge

Standard MIR datasets (like GTZAN) consist of clean, studio-quality recordings. This project introduces an extreme "Messy Mashup" variant where the model must classify the dominant genre of tracks that have been deliberately corrupted. 

We were provided with **1,000 clean music tracks** split into isolated instrument stems (vocals, drums, bass, other) and **2,000 environmental noise clips** from the ESC-50 dataset. The test set consisted of **3,020 pre-corrupted mashups**. To train the model, we had to programmatically synthesize our own training data to simulate the unknown corruption distribution of the test set.

## 🧬 Data Synthesis & Augmentation Pipeline

To prevent catastrophic overfitting and bridge the domain gap between clean stems and the noisy test set, a highly aggressive synthetic audio generation pipeline was engineered using `librosa` and PyTorch:

*   **Cross-Stem Blending:** Dynamically blending random stems (e.g., drums from Track A, vocals from Track B) within the same genre at randomized amplitude weights (0.3x to 1.5x) to prevent track-specific memorization.
*   **ESC-50 Noise Injection:** Mathematically calculating signal power to inject random ESC-50 environmental noise (sirens, rain, etc.) at varying Signal-to-Noise Ratios (SNR) between 0dB and 15dB.
*   **Phase-Vocoder Time Stretching:** Altering the tempo of the audio without shifting the pitch (0.88x to 1.12x) to build rhythm-invariant embeddings.
*   **SpecAugment:** Applying severe time and frequency masking directly to the generated Mel-Spectrogram tensors to force global contextual learning.

## 🧠 Triple-Ensemble Architecture

No single model architecture could untangle the overlapping frequencies and phase distortions perfectly. A dynamically weighted triple-ensemble was designed to leverage different architectural inductive biases:

1.  **Audio Spectrogram Transformer (AST):** 
    *   *Role:* Global feature correlation. Treats the 1024x128 Mel-Spectrogram as a 2D image, utilizing Vision Transformer (ViT) self-attention and positional embeddings to capture long-range harmonic dependencies. 
2.  **MERT (Music Expression Representation Transfer):**
    *   *Role:* Raw acoustic phase capture. An acoustic representation model based on HuBERT that processes raw 1D audio waveforms, inherently capturing micro-temporal nuances and phase information lost during standard STFT extraction.
3.  **EfficientNet-B2:**
    *   *Role:* Localized visual pattern extraction. Scaled resolution, depth, and width perfectly to extract localized spectrogram patterns without introducing a massive computational bottleneck during inference.

## 🚀 Inference Strategy: Test-Time Augmentation (TTA)

To stabilize the final predictions across the highly volatile test distribution, an aggressive **15-pass Test-Time Augmentation** strategy was implemented:
*   **11 Clean Passes:** Utilizing distinct temporal offsets (ranging from 0.0s to 25.0s) to scan the entire track.
*   **4 Noise-Injected Passes:** Re-injecting random noise at 8dB and 12dB SNR during inference. This acts as a powerful regularizer, ensuring anomalous spikes in the original test audio are smoothed out when the softmax probabilities are averaged.

## 🏆 Results

The dynamically weighted probability fusion of the three models successfully penetrated the noise, achieving top-tier performance on the unseen Kaggle dataset:

*   **Public Test Score:** `0.95595`
*   **Private Test Score:** `0.96093` (Macro F1)

## 🌐 Deployment

The trained model weights and inference pipeline were packaged and deployed as a real-time web application on **Hugging Face Spaces** using the Gradio framework, featuring a Dual-Mode interface (Fast Mode vs. Max Accuracy Ensemble).
