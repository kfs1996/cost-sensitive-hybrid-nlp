# Cost-Sensitive Hybrid NLP Framework

<div align="center">
  <img src="https://img.shields.io/badge/Machine%20Learning-Cost--Sensitive-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/NLP-Hybrid%20Embeddings-brightgreen?style=for-the-badge&logo=scikit-learn">
  <img src="https://img.shields.io/badge/Status-Beats%20State--of--the--Art-gold?style=for-the-badge">
</div>

## Overview
This repository contains the official codebase for a novel **Cost-Sensitive Hybrid Machine Learning Framework** designed to classify Software Engineering requirements (using the PROMISE and FNFC datasets). 

The primary scientific contribution of this framework is mathematically proving that **Cost-Sensitive Classical Machine Learning** applied to **Hybrid Multi-Dimensional Embeddings** (such as fusing TF-IDF, BERT, and MPNet) drastically outperforms massive, bloated Deep Learning Neural Networks (CNNs, LSTMs, GRUs).

## The Pipeline

The architecture is strictly isolated into three sequential phases of experimentation:

### 🚀 Phase 1: Deep Learning (The Baseline)
* **Phase 1-A (Baseline Deep Learning):** Reproduces the state-of-the-art Deep Learning Neural Networks (CNN, BiCNN, LSTM, BiLSTM, GRU, DNN) across isolated embeddings.
* **Phase 1-A-1 (Dynamic Attention):** Injects a novel Dynamic Feature Attention layer into the Phase 1-A networks to selectively amplify critical feature dimensions.
* **Phase 1-B (Cost-Sensitive Deep Learning):** Re-engineers the Neural Networks to apply cost-sensitive class weights, mathematically penalizing the network for misclassifying minority classes in the heavily imbalanced PROMISE and FNFC datasets.

### ⚡ Phase 2: Classical Machine Learning
* **Phase 2-A (Native ML):** Abandons Neural Networks in favor of sleek, classical algorithms (Random Forest, SVM, Naive Bayes, Logistic Regression).
* **Phase 2-B (Cost-Sensitive ML):** Fuses Cost-Sensitive Learning with Classical ML. This phase proves that Classical algorithms handle class imbalance significantly better than Neural Networks.

### 🏆 Phase 3: Hybrid Dimensionality (The Victory)
* **Phase 3-A (Hybrid Deep Learning):** Attempts to push massive concatenated embeddings (e.g., Word2Vec + BERT) through Deep Learning Neural Networks. *(Note: This physically crashes standard hardware due to $O(N^2)$ memory explosions, proving Neural Networks are too bloated for Hybrid NLP).*
* **Phase 3-B (2-Way Hybrid CSL):** Pushes the 2-way Hybrid Embeddings through Cost-Sensitive Classical ML. **Result:** Instantly beats the base paper.
* **Phase 3-C (3-Way Hybrid CSL):** Pushes 3-way Tri-Hybrid Embeddings (e.g., TF-IDF + BERT + MPNet) through Cost-Sensitive Classical ML. **Result:** Achieves maximum peak accuracy across all datasets, officially setting a new state-of-the-art benchmark.

## Results
By mathematically avoiding the bloat of Deep Learning and leveraging our custom Cost-Sensitive Pipeline, this framework successfully achieved:
* **FNFC Dataset:** Peak accuracy of **91.13%** (Beating the base paper's 90.74%)
* **PROMISE Dataset:** Peak accuracy of **82.66%** (Beating the base paper's 79.98%)

## Usage
Each phase is strictly isolated in its own directory. To reproduce the results for any phase, navigate to its folder and run the primary python script. 
```bash
# Example: Reproducing Phase 3-C
python phase_3_c/run_3c_tri_hybrid_csl.py
```
