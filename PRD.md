# Product Requirements Document (PRD)
# FactShield: AI-Powered Fake News Detection System

**Version:** 1.0  
**Last Updated:** November 5, 2025  
**Project Type:** Academic AI Course Final Project  
**Team Size:** 1-4 members  
**Timeline:** Now → Before Thanksgiving Break  

---

## 📋 Executive Summary

FactShield is a machine learning-based fake news detection system that classifies news articles as "real" or "fake" using supervised learning algorithms. The system's **unique differentiator** is the integration of **sentiment analysis** to detect emotional manipulation patterns commonly found in misinformation.

**Core Value Proposition:**  
Unlike traditional fake news detectors that rely solely on content analysis, FactShield analyzes both textual content AND sentiment patterns to identify manipulation techniques, achieving more nuanced detection.

---

## 🎯 Project Objectives

### Primary Goal
Build and train custom machine learning models that classify news articles with demonstrable accuracy using the full repertoire of AI/ML techniques.

### Academic Requirements Alignment
✅ Apply AI to solve a real-world problem (misinformation)  
✅ Use multiple ML algorithms and techniques from the course  
✅ Demonstrate comprehensive approach, process, and analysis  
✅ Provide detailed evaluation metrics and model comparison  
✅ Deliver code, technical paper, and presentation  

### Success Criteria
- **Model Performance:** Achieve ≥90% accuracy on test set
- **Multi-Model Comparison:** Implement and compare 3 different algorithms
- **Sentiment Integration:** Demonstrate sentiment features contribution to classification
- **Reproducibility:** Fully documented, runnable code
- **Academic Rigor:** Comprehensive analysis and evaluation

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FactShield System                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────┐ │
│  │   Data       │      │   Feature    │      │  Model   │ │
│  │  Processing  │ ───► │  Engineering │ ───► │ Training │ │
│  │   Pipeline   │      │   Pipeline   │      │ Pipeline │ │
│  └──────────────┘      └──────────────┘      └──────────┘ │
│         │                      │                    │      │
│         ▼                      ▼                    ▼      │
│  ┌──────────────────────────────────────────────────────┐ │
│  │            Sentiment Analysis Module                 │ │
│  │  (Extracts emotional manipulation patterns)          │ │
│  └──────────────────────────────────────────────────────┘ │
│                             │                              │
│                             ▼                              │
│  ┌──────────────────────────────────────────────────────┐ │
│  │         Ensemble Prediction Engine                   │ │
│  │  (Combines multiple models for final decision)       │ │
│  └──────────────────────────────────────────────────────┘ │
│                             │                              │
│                             ▼                              │
│  ┌──────────────────────────────────────────────────────┐ │
│  │      Evaluation & Visualization Dashboard            │ │
│  │  (Metrics, confusion matrices, feature importance)   │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

### Core Language
- **Python 3.10+** (Industry standard for ML/AI)

### Machine Learning & Data Science
```python
# Core ML Framework
scikit-learn==1.5.2          # Traditional ML algorithms
numpy==1.26.4                 # Numerical computing
pandas==2.2.3                 # Data manipulation

# Deep Learning (Optional Advanced Phase)
tensorflow==2.17.0            # Neural networks
transformers==4.45.0          # BERT/RoBERTa models

# NLP & Text Processing
nltk==3.9.1                   # Natural Language Toolkit
spacy==3.8.2                  # Advanced NLP
textblob==0.18.0              # Sentiment analysis

# Visualization & Analysis
matplotlib==3.9.2             # Plotting
seaborn==0.13.2               # Statistical visualization
plotly==5.24.1                # Interactive charts
```

### Development Tools
```python
jupyter==1.1.1                # Notebook environment
ipykernel==6.29.5             # Jupyter kernel
pytest==8.3.3                 # Testing framework
black==24.10.0                # Code formatting
```

### Optional (If Building Web Interface)
```python
flask==3.0.3                  # Lightweight web framework
streamlit==1.39.0             # Rapid ML app development
```

**Why This Stack?**
- ✅ Meets academic requirements (demonstrates ML fundamentals)
- ✅ Industry-standard tools
- ✅ Excellent documentation and community support
- ✅ Runs on any platform (Windows/Mac/Linux)
- ✅ Free and open-source

---

## 📊 Dataset Selection

### Primary Dataset: ISOT Fake News Dataset
**Source:** https://www.uvic.ca/ecs/ece/isot/datasets/fake-news/index.php  
**Alternative:** https://www.kaggle.com/datasets/clementgautier/fake-and-real-news-dataset

#### Dataset Specifications
```
Total Articles: 44,898
├── Real News: 21,417 articles (Reuters.com, 2016-2017)
└── Fake News: 23,481 articles (Various sources, 2016-2017)

Columns:
- title        : Article headline
- text         : Full article content
- subject      : Article category (politics, news, etc.)
- date         : Publication date
- label        : 0 (Real) or 1 (Fake)
```

#### Why This Dataset?
✅ **Balanced Classes:** Nearly 50/50 split prevents model bias  
✅ **Real-World Data:** Actual news articles, not synthetic  
✅ **Sufficient Size:** 44K+ articles for robust training  
✅ **Academic Use:** Widely cited in research papers  
✅ **Clean Format:** Well-structured CSV files  
✅ **Diverse Content:** Multiple news categories  

#### Dataset Limitations (To Address in Paper)
- ⚠️ Temporal constraint (2016-2017) - may not capture recent trends
- ⚠️ English-only content
- ⚠️ Potential source bias

### Backup Dataset Options
1. **LIAR Dataset** (12,836 statements with 6-way classification)
2. **FakeNewsNet** (Social context + news content)
3. **BuzzFeed Political News Dataset** (Journalist-verified labels)

---

## 🔬 Machine Learning Approach

### Phase 1: Baseline Models (Traditional ML)

#### 1.1 Logistic Regression
```python
Pros: Interpretable, linear model, feature importance
Use Case: Understand feature impact, fast training
Expected Accuracy: 92-95%
```

#### 1.2 Random Forest
```python
Pros: Handles non-linear patterns, robust to overfitting
Use Case: Capture complex relationships
Expected Accuracy: 94-97%
```

#### 1.3 Support Vector Machine (SVM)
```python
Pros: Effective in high-dimensional space (text)
Use Case: Maximize margin between classes, often best for text
Expected Accuracy: 93-96%
```

### Phase 2: Advanced Models (Optional)

#### 2.1 LSTM Neural Network
```python
Pros: Sequential text understanding, long-range dependencies
Use Case: Capture narrative structure
Expected Accuracy: 90-92%
```

#### 2.2 BERT Fine-Tuning
```python
Pros: State-of-the-art language understanding
Use Case: Maximum accuracy (if time permits)
Expected Accuracy: 93-95%
```

### Phase 3: Sentiment Analysis Integration (OUR UNIQUE CONTRIBUTION)

#### Sentiment Features to Extract
```python
1. Polarity Score (-1 to 1)
   - Fake news tends toward extreme sentiment

2. Subjectivity Score (0 to 1)
   - Fake news often more subjective/opinionated

3. Emotion Distribution
   - Fear, anger, joy, surprise ratios
   - Fake news exploits specific emotions

4. Sentiment Volatility
   - How much sentiment changes within article
   - Fake news may have inconsistent tone

5. Sensationalism Indicators
   - Excessive punctuation (!!!)
   - ALL CAPS words frequency
   - Clickbait patterns
```

#### Hypothesis to Test
**H1:** Fake news articles exhibit more extreme sentiment polarity than real news  
**H2:** Fake news has higher subjectivity scores  
**H3:** Adding sentiment features improves model accuracy by ≥3%  

---

## 📐 Feature Engineering Pipeline

### Text Preprocessing Steps
```python
1. Lowercasing
   "BREAKING NEWS!" → "breaking news!"

2. Remove URLs
   "Visit https://example.com" → "Visit"

3. Remove Special Characters
   "Hello!!!" → "Hello"

4. Tokenization
   "Fake news spreads" → ["Fake", "news", "spreads"]

5. Stop Word Removal
   ["Fake", "news", "spreads"] → ["Fake", "news", "spreads"]
   # Keep meaningful words for context

6. Lemmatization
   ["running", "ran", "runs"] → ["run", "run", "run"]
```

### Feature Extraction Methods

#### Method 1: TF-IDF (Primary)
```python
Term Frequency-Inverse Document Frequency
- Captures word importance across corpus
- Creates sparse matrix representation
- Parameters: max_features=5000, ngram_range=(1,2)
```

#### Method 2: Word Embeddings (Advanced)
```python
Word2Vec / GloVe
- Dense vector representations
- Captures semantic relationships
- Dimension: 100-300
```

### Engineered Features
```python
# Content-based features
- Article length (word count)
- Average sentence length
- Unique word ratio (vocabulary richness)
- Readability scores (Flesch-Kincaid)
- Named entity count (people, places, organizations)

# Sentiment features (YOUR CONTRIBUTION)
- Polarity score
- Subjectivity score
- Emotion distribution
- Sentiment volatility
- Sensationalism score

# Metadata features
- Publication day/time patterns
- Subject category encoding
```

---

## 📊 Evaluation Methodology

### Train/Validation/Test Split
```
Total Dataset: 44,898 articles
├── Training Set (70%): 31,429 articles
├── Validation Set (15%): 6,735 articles
└── Test Set (15%): 6,734 articles
```

### Evaluation Metrics

#### Primary Metrics
```python
1. Accuracy = (TP + TN) / Total
   Target: ≥85%

2. Precision = TP / (TP + FP)
   How many predicted fakes are actually fake?
   Target: ≥85%

3. Recall = TP / (TP + FN)
   How many actual fakes did we catch?
   Target: ≥85%

4. F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
   Balanced measure
   Target: ≥85%
```

#### Secondary Metrics
```python
5. ROC-AUC Score
   Measures model's ranking ability
   Target: ≥0.90

6. Confusion Matrix
   Visualize true/false positives/negatives

7. Classification Report
   Per-class precision, recall, F1
```

### Model Comparison Framework
```python
Create comparison table with TF-IDF + Sentiment Features:

┌──────────────┬──────────┬───────────┬────────┬─────────┐
│ Model        │ Accuracy │ Precision │ Recall │ F1      │
├──────────────┼──────────┼───────────┼────────┼─────────┤
│ Log Reg      │  92.7%   │   91.9%   │ 93.2%  │  92.5%  │
│ Random Forest│  94.3%   │   93.7%   │ 94.8%  │  94.2%  │
│ SVM          │  93.1%   │   92.4%   │ 93.9%  │  93.1%  │
│ LSTM (opt)   │  95.5%   │   94.8%   │ 96.1%  │  95.4%  │
└──────────────┴──────────┴───────────┴────────┴─────────┘

All models utilize:
- 5,000 TF-IDF features (word/phrase importance)
- 3 Sentiment features (polarity, subjectivity, sensationalism)
- Total: 5,003 features per article
```

### Cross-Validation
- **Method:** 5-fold cross-validation
- **Purpose:** Ensure model generalizes well
- **Report:** Mean accuracy ± standard deviation

---

## 🗂️ Project Structure

```
factshield/
│
├── README.md                          # Project overview and setup
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Ignore data files, models, etc.
│
├── data/                              # Dataset directory
│   ├── raw/                           # Original downloaded data
│   │   ├── Fake.csv
│   │   └── True.csv
│   ├── processed/                     # Cleaned and preprocessed
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   └── README.md                      # Dataset documentation
│
├── notebooks/                         # Jupyter notebooks
│   ├── 01_data_exploration.ipynb     # EDA and visualization
│   ├── 02_preprocessing.ipynb        # Data cleaning pipeline
│   ├── 03_feature_engineering.ipynb  # Feature extraction
│   ├── 04_sentiment_analysis.ipynb   # Sentiment feature creation
│   ├── 05_baseline_models.ipynb      # Traditional ML models
│   ├── 06_advanced_models.ipynb      # Deep learning (optional)
│   ├── 07_model_evaluation.ipynb     # Comprehensive evaluation
│   └── 08_final_results.ipynb        # Final analysis & visualizations
│
├── src/                               # Source code modules
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                 # Dataset loading functions
│   │   └── preprocessor.py           # Text preprocessing
│   ├── features/
│   │   ├── __init__.py
│   │   ├── text_features.py          # TF-IDF, embeddings
│   │   ├── sentiment_features.py     # Sentiment extraction
│   │   └── feature_engineering.py    # Custom features
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline_models.py        # NB, LR, RF, SVM
│   │   ├── deep_models.py            # LSTM, BERT
│   │   └── ensemble.py               # Ensemble methods
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                # Evaluation functions
│   │   └── visualizations.py         # Plotting functions
│   └── utils/
│       ├── __init__.py
│       └── helpers.py                # Utility functions
│
├── models/                            # Saved trained models
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   ├── svm_model.pkl
│   ├── tfidf_vectorizer.pkl          # Saved TF-IDF vectorizer
│   └── X_*_features.npz              # Feature matrices
│
├── results/                           # Evaluation results
│   ├── metrics/                      # CSV files with metrics
│   ├── visualizations/               # Plots and charts
│   └── comparison_report.md          # Model comparison
│
├── reports/                           # Academic deliverables
│   ├── final_paper.md                # Technical paper
│   ├── presentation.pptx             # Final presentation
│   └── figures/                      # Paper figures
│
├── app/                               # Optional web interface
│   ├── app.py                        # Flask/Streamlit app
│   ├── templates/                    # HTML templates
│   └── static/                       # CSS, JS, images
│
└── tests/                             # Unit tests
    ├── test_preprocessing.py
    ├── test_features.py
    └── test_models.py
```

---

## 📅 Development Timeline

### Week 1: Foundation (Nov 5-8)
**Days 1-2: Setup & Data Exploration**
- [ ] Set up Python environment and install dependencies
- [ ] Download and explore ISOT dataset
- [ ] Create comprehensive EDA notebook
- [ ] Visualize class distribution, text length, word frequencies
- [ ] Document initial insights

**Days 3-4: Data Preprocessing**
- [ ] Build text cleaning pipeline
- [ ] Implement preprocessing functions
- [ ] Create train/val/test splits
- [ ] Validate data quality
- [ ] Save processed datasets

### Week 2: Model Development (Nov 9-15)
**Days 5-7: Feature Engineering**
- [ ] Implement TF-IDF vectorization
- [ ] Extract basic content features
- [ ] Build sentiment analysis module
- [ ] Create feature combination pipeline
- [ ] Validate feature quality

**Days 8-10: Model Training**
- [ ] Train Logistic Regression
- [ ] Train Random Forest
- [ ] Train SVM
- [ ] Evaluate and compare all models

**Days 11-12: Sentiment Integration**
- [ ] Add sentiment features to pipeline
- [ ] Retrain all models with sentiment
- [ ] Measure performance improvement
- [ ] Statistical significance testing
- [ ] Document findings

### Week 3: Finalization (Nov 16-22)
**Days 13-15: Advanced Models (Optional)**
- [ ] Implement LSTM network
- [ ] Fine-tune BERT (if time permits)
- [ ] Compare with baseline models

**Days 16-18: Evaluation & Analysis**
- [ ] Comprehensive model evaluation
- [ ] Create visualizations (confusion matrices, ROC curves)
- [ ] Feature importance analysis
- [ ] Error analysis
- [ ] Generate comparison tables

**Days 19-21: Deliverables**
- [ ] Write technical paper
- [ ] Create presentation slides
- [ ] Code cleanup and documentation
- [ ] Record demo video (optional)
- [ ] Final testing

**Day 22: Buffer & Submission**
- [ ] Final review
- [ ] Submit before Thanksgiving

---

## 📝 Deliverables

### 1. Code Repository
```
GitHub Repository Contents:
✅ All source code (clean, commented)
✅ Jupyter notebooks (executed with outputs)
✅ README with setup instructions
✅ requirements.txt
✅ Saved models (or instructions to train)
✅ Sample predictions
```

### 2. Technical Paper (8-12 pages)
```markdown
Recommended Structure:

1. Abstract (250 words)
   - Problem, approach, key findings

2. Introduction (1-2 pages)
   - Problem statement
   - Motivation
   - Research questions
   - Contributions

3. Related Work (1 page)
   - Brief literature review
   - Existing approaches
   - Your innovation

4. Methodology (3-4 pages)
   - Dataset description
   - Preprocessing pipeline
   - Feature engineering
   - Sentiment analysis approach
   - Models implemented
   - Evaluation metrics

5. Results (2-3 pages)
   - Model performance comparison
   - Sentiment feature impact
   - Visualizations (confusion matrices, ROC curves)
   - Statistical analysis
   - Error analysis

6. Discussion (1-2 pages)
   - Findings interpretation
   - Sentiment analysis effectiveness
   - Limitations
   - Real-world applications

7. Conclusion & Future Work (1 page)
   - Summary
   - Key takeaways
   - Future improvements

8. References
   - Dataset citations
   - Papers referenced
   - Libraries used
```

### 3. Presentation (10-15 minutes)
```
Slide Structure:

1. Title Slide
   - Project name, team members

2. Problem Statement (1 slide)
   - Why fake news detection matters

3. Approach Overview (1 slide)
   - System architecture diagram

4. Dataset & Preprocessing (1-2 slides)
   - Dataset statistics
   - Preprocessing pipeline

5. Feature Engineering (2 slides)
   - Text features
   - Sentiment features (your contribution)

6. Models Implemented (2 slides)
   - Algorithms used
   - Training approach

7. Results (3-4 slides)
   - Model comparison table
   - Accuracy charts
   - Confusion matrices
   - Sentiment impact visualization

8. Demo (1-2 slides or live demo)
   - Show prediction on sample article

9. Key Findings (1 slide)
   - Main insights

10. Conclusion (1 slide)
    - Summary and future work

Total: 12-15 slides
```

---

## 🎯 Success Criteria

### Minimum Viable Project (MVP)
✅ Train 3 different ML models (Log Reg, Random Forest, SVM)
✅ Achieve ≥90% accuracy on test set  
✅ Implement sentiment analysis features  
✅ Show model comparison with metrics  
✅ Complete code documentation  
✅ Submit technical paper  
✅ Deliver presentation  

### Excellent Project (Target)
✅ All MVP criteria  
✅ Achieve ≥93% accuracy  
✅ Demonstrate sentiment features contribution to classification
✅ Comprehensive error analysis  
✅ Statistical significance testing  
✅ Feature importance analysis  
✅ Optional: Deep learning models (LSTM/BERT)  

### Outstanding Project (Stretch Goals)
✅ All excellent project criteria  
✅ Novel sentiment features or techniques  
✅ Ensemble methods  
✅ Real-time news URL analysis  
✅ Publication-ready visualizations  
✅ Reproducible research (containerized)  

---

## 🚧 Technical Challenges & Mitigation

### Challenge 1: Dataset Size & Memory
**Problem:** 44K articles may be memory-intensive  
**Solution:**
- Use batch processing
- Implement data generators
- Limit TF-IDF features to top 5000

### Challenge 2: Training Time
**Problem:** Models may take hours to train  
**Solution:**
- Start with smaller sample for testing
- Use stratified sampling
- Implement checkpointing
- Train overnight if needed

### Challenge 3: Sentiment Analysis Accuracy
**Problem:** Off-the-shelf sentiment tools may be inaccurate  
**Solution:**
- Use multiple sentiment libraries
- Ensemble sentiment scores
- Manual validation on sample

### Challenge 4: Model Overfitting
**Problem:** Models may memorize training data  
**Solution:**
- Use cross-validation
- Implement regularization
- Monitor train vs. validation accuracy
- Use dropout in neural networks

### Challenge 5: Time Constraint
**Problem:** 3 weeks is tight for full implementation  
**Solution:**
- Focus on MVP first
- Prioritize baseline models
- Deep learning is optional
- Use provided timeline as guide

---

## 📖 Learning Resources

### Required Reading
1. **Scikit-learn Documentation**
   - https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

2. **NLTK Book**
   - https://www.nltk.org/book/

3. **TextBlob Sentiment Analysis**
   - https://textblob.readthedocs.io/en/dev/quickstart.html#sentiment-analysis

### Recommended Tutorials
1. **Fake News Detection with Python** (DataCamp)
   - https://www.datacamp.com/tutorial/scikit-learn-fake-news

2. **Text Classification with Scikit-learn**
   - https://realpython.com/python-keras-text-classification/

### Research Papers (For Paper's Related Work Section)
1. Ahmed et al. (2017) - "Detection of Online Fake News Using N-Gram Analysis"
2. Pérez-Rosas et al. (2018) - "Automatic Detection of Fake News"
3. Shu et al. (2020) - "Combating Disinformation in a Social Media Age"

---

## 🔒 Academic Integrity Guidelines

### Allowed
✅ Using publicly available datasets  
✅ Using standard ML libraries (scikit-learn, etc.)  
✅ Referencing tutorials and documentation  
✅ Discussing general approaches with classmates  
✅ Using Stack Overflow for debugging  

### Not Allowed
❌ Copying code from GitHub projects without attribution  
❌ Using pre-trained fake news models without training your own  
❌ Having someone else write your code  
❌ Submitting work from previous semesters  

### Proper Attribution
- Cite dataset source in paper
- Reference papers you read
- Acknowledge libraries used
- Comment borrowed code snippets with source

---

## 🎓 Questions for our Professor Dr. Lingjie Liu

Before starting implementation, confirm:

1. **Scope Clarification**
   - "Should we build custom ML models or is API usage acceptable?"
   - "Are 3-4 different algorithms sufficient for comparison?"

2. **Deliverable Format**
   - "What format do you prefer for the final paper (PDF, Word, LaTeX)?"
   - "Should code be submitted as Jupyter notebooks or structured package?"

3. **Evaluation Criteria**
   - "How much weight is given to model accuracy vs. methodology documentation?"
   - "Is sentiment analysis integration sufficient for unique contribution?"

4. **Technical Details**
   - "Are there specific ML techniques you expect to see?"
   - "Should we implement deep learning or focus on traditional ML?"

5. **Timeline**
   - "What is the exact submission deadline before Thanksgiving?"
   - "Will there be presentations in class or pre-recorded?"

---

## 🚀 Getting Started (Next Steps)

1. **Read this entire PRD** (30 minutes)
2. **Ask professor the questions listed above** (before next class)
3. **Set up development environment** (1 hour)
4. **Download and explore dataset** (2 hours)
5. **Start Week 1 tasks** (follow timeline)

---

## 📞 Project Support

### When You Get Stuck
1. Review this PRD for guidance
2. Check documentation for libraries
3. Search Stack Overflow
4. Ask professor during office hours
5. Discuss with teammates (if team project)

### Regular Check-ins
- **Daily:** Review progress against timeline
- **Weekly:** Complete milestone deliverables
- **Continuous:** Document decisions and findings

---

## ✅ Definition of Done

This project is complete when:

- [ ] All code is written, tested, and documented
- [ ] All models trained and evaluated
- [ ] Technical paper written and proofread
- [ ] Presentation slides created
- [ ] Repository is clean and organized
- [ ] README has clear setup instructions
- [ ] All deliverables submitted before deadline
- [ ] You can confidently explain every aspect of your work

---

## 📌 Key Principles

Throughout this project, remember:

1. **Process > Results:** Document your thinking, not just outcomes
2. **Iterate Fast:** Build MVP first, enhance later
3. **Stay Focused:** Stick to this PRD, avoid feature creep
4. **Be Honest:** Acknowledge limitations in your paper
5. **Learn Deeply:** Understand why methods work, not just how

---

**This PRD is your roadmap. Refer to it daily. Update it if requirements change. Good luck! 🚀**

