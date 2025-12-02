# 🛡️ FactShield: AI-Powered Fake News Detection System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.2-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-99.70%25-brightgreen.svg)](#results)

> **Academic Project:** Artificial Intelligence Course (ITEC-4700) Final Project  
> **Institution:** Fall 2025  
> **Authors:** Bereket Takiso (Lead), Puru Mukherjee, Albert Austin, Fizza Haider  
> **Group:** Group 1

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Results](#-key-results)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Results](#-results)
- [Authors](#-authors)
- [References](#-references)

---

## 🎯 Overview

**FactShield** is a machine learning-based fake news detection system that achieves **99.70% accuracy** in classifying news articles as "real" or "fake." 

### What Makes FactShield Unique?

Unlike traditional fake news detectors that rely solely on word frequencies, FactShield integrates **sentiment analysis** to detect emotional manipulation patterns:

- **Fake news is 20% MORE subjective** than real news (statistically proven, p < 0.001)
- Combines **5,000 TF-IDF features** with **3 sentiment features**
- Only **20 errors** out of 6,717 test articles

---

## 🏆 Key Results

| Metric | Score |
|--------|-------|
| **Accuracy** | **99.70%** |
| **Precision** | 99.75% |
| **Recall** | 99.63% |
| **F1-Score** | 99.69% |
| **Error Rate** | 0.30% (20/6,717 articles) |
| **Prediction Speed** | 171,716 articles/second |

### Model Comparison

| Model | Accuracy | F1-Score | Training Time |
|-------|----------|----------|---------------|
| **SVM** ⭐ | **99.76%** | **99.77%** | **2.25s** |
| Random Forest | 99.66% | 99.67% | 21.92s |
| Logistic Regression | 99.30% | 99.33% | 7.12s |

---

## ✨ Features

- **Multi-Model Approach:** Implements 3 ML algorithms (Logistic Regression, Random Forest, SVM)
- **Novel Sentiment Analysis:** Detects emotional manipulation patterns (polarity, subjectivity, sensationalism)
- **Statistical Validation:** All sentiment features statistically significant (p < 0.001)
- **Production-Ready:** Fast inference (171K articles/second), small model size (0.2 MB)
- **No Overfitting:** Only 0.08% difference between validation and test performance
- **Comprehensive Evaluation:** Detailed metrics, confusion matrices, error analysis
- **Reproducible Research:** Fully documented Jupyter notebooks

---

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- 8GB+ RAM recommended

### Step 1: Clone the Repository

```bash
git clone https://github.com/btakiso/FactShield.git
cd FactShield
```

### Step 2: Create Virtual Environment (Optional but Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('omw-1.4')"
```

### Step 5: Download Dataset

1. Visit: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
2. Download the dataset (requires free Kaggle account)
3. Extract `Fake.csv` and `True.csv`
4. Place both files in `data/raw/` directory

---

## 📖 Usage

### Running the Notebooks

```bash
jupyter notebook
```

**Run notebooks in order:**

1. `01_quick_start.ipynb` - Data exploration and loading
2. `02_data_preprocessing.ipynb` - Text cleaning and train/val/test split
3. `03_feature_engineering.ipynb` - TF-IDF + Sentiment feature extraction
4. `04_model_training.ipynb` - Train 3 ML models
5. `05_final_evaluation.ipynb` - Final test results and error analysis

### Quick Test

After running all notebooks, you should see:

```
✅ SVM Test Accuracy: 99.70%
✅ Only 20 errors out of 6,717 articles
✅ F1-Score: 99.69%
```

---

## 📁 Project Structure

```
FactShield/
├── notebooks/                    # Jupyter notebooks (run these!)
│   ├── 01_quick_start.ipynb     # Data exploration
│   ├── 02_data_preprocessing.ipynb  # Text cleaning
│   ├── 03_feature_engineering.ipynb # TF-IDF + Sentiment
│   ├── 04_model_training.ipynb  # Model training
│   └── 05_final_evaluation.ipynb    # Final results
│
├── data/
│   ├── raw/                     # Original dataset (download from Kaggle)
│   └── processed/               # Cleaned train/val/test splits
│
├── models/                      # Trained models and features
│   ├── svm_model.pkl           # Best model (SVM)
│   ├── tfidf_vectorizer.pkl    # TF-IDF vectorizer
│   └── *.npz, *.npy            # Feature matrices and labels
│
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── PRD.md                       # Product Requirements Document
```

---

## 🔬 Methodology

### Data Pipeline

```
Raw Data (44,898 articles)
    ↓
Text Preprocessing (cleaning, tokenization, lemmatization)
    ↓
Feature Engineering
    ├── TF-IDF Vectorization (5,000 features)
    └── Sentiment Analysis (3 features)
    ↓
Combined Features (5,003 per article)
    ↓
Model Training (SVM, Random Forest, Logistic Regression)
    ↓
Evaluation (99.70% accuracy on test set)
```

### Sentiment Analysis Features

| Feature | Description | Key Finding |
|---------|-------------|-------------|
| **Polarity** | Positive vs. negative sentiment (-1 to +1) | Fake news slightly more positive |
| **Subjectivity** | Opinion vs. fact-based (0 to 1) | **Fake news 20% MORE subjective** |
| **Sensationalism** | Exclamation marks, ALL CAPS usage | Fake news more dramatic |

All differences statistically significant (p < 0.001)

---

## 📊 Results

### Final Test Performance (6,717 unseen articles)

```
Accuracy:   99.70%  (6,697 correct / 6,717 total)
Precision:  99.75%
Recall:     99.63%
F1-Score:   99.69%

Confusion Matrix:
                 Predicted
              Fake    Real
Actual Fake   3,496      8    ← Only 8 false positives
       Real     12   3,201    ← Only 12 false negatives

Total Errors: 20 out of 6,717 (0.30% error rate)
```

### No Overfitting

| Metric | Validation | Test | Difference |
|--------|-----------|------|------------|
| F1-Score | 99.77% | 99.69% | **0.08%** |

The minimal difference proves excellent generalization!

---

## 👥 Authors

| Name | Role |
|------|------|
| **Bereket Takiso** | Project Lead - Architecture, Implementation, All Notebooks, Documentation |
| Puru Mukherjee | Data preprocessing validation, Code review |
| Albert Austin | Literature review, Project planning |
| Fizza Haider | Paper formatting, Presentation review |

---

## 📚 References

### Dataset
- Kaggle Fake & Real News Dataset: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

### Key Papers
- Pérez-Rosas, V., et al. (2018). "Automatic Detection of Fake News"
- Shu, K., et al. (2020). "Combating Disinformation in a Social Media Age"
- Zhou, X., & Zafarani, R. (2020). "A Survey of Fake News"

### Libraries
- scikit-learn: https://scikit-learn.org/
- NLTK: https://www.nltk.org/
- TextBlob: https://textblob.readthedocs.io/
- pandas: https://pandas.pydata.org/

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Professor Dr. Lingjie Liu for course guidance
- Kaggle for hosting the dataset
- ISOT Research Lab for creating the original dataset

---

**⭐ If you found this project useful, please give it a star!**

---

*Last Updated: December 2025*
