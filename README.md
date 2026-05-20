<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=2,12,24&height=220&section=header&text=🗞️%20FAKE%20NEWS%20DETECTOR%20🗞️&fontSize=38&fontColor=fff&animation=twinkling&fontAlignY=38&desc=Real%20or%20Fake%3F%20The%20AI%20Knows.&descAlignY=58&descSize=16"/>

# ⚡ Teaching Machines to Spot Lies ⚡

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Pandas](https://img.shields.io/badge/Pandas-Data_Wrangling-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-NLP_Pipeline-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-4C72B0?style=for-the-badge)](https://seaborn.pydata.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![Status](https://img.shields.io/badge/Status-🔥_COMPLETE-brightgreen?style=for-the-badge)](#)

<br/>

---

### 💬 *"Misinformation spreads faster than truth.*
### *This AI was built to close that gap."*

---

> 📰 **Feed it a news article.**
> 🤖 **NLP + ML dissects every word.**
> ✅ **REAL** or ❌ **FAKE** — verdict delivered instantly.

---

</div>

## 🚨 Why This Project Matters

Every day, **millions of fake news articles** circulate on the internet — swaying elections, spreading fear, and destroying reputations. Traditional fact-checking is slow. Human reviewers are limited.

This project answers one question: **Can a machine learn to detect a lie?**

```
📰 News Article Text  →  🧠 NLP Feature Extraction  →  🤖 ML Classifier  →  ✅ REAL / ❌ FAKE
```

The answer: **Yes. And it does it at scale.**

---

## 📊 The Dataset

```
📂 News.csv
 ┣ 📰 title     →  Headline of the article
 ┣ 📝 text      →  Full body of the article
 ┣ 🏷️  subject  →  News category (e.g., Politics, News)
 ┣ 📅 date      →  Publication date
 ┗ 🎯 label     →  REAL or FAKE (target variable)
```

> Dataset includes real political news articles alongside fabricated ones — making this a genuine, high-stakes classification challenge.

---

## 🧠 The NLP + ML Pipeline

```
╔══════════════════════════════════════════════════════════════╗
║            FAKE NEWS DETECTION — HOW IT WORKS               ║
╠══════════════════════════════════════════════════════════════╣
║  📰 Raw Article (title + text)                               ║
║       ↓                                                      ║
║  🧹 Text Preprocessing   →  Clean, lowercase, strip noise    ║
║       ↓                                                      ║
║  🔢 Feature Extraction   →  TF-IDF Vectorization             ║
║       ↓                                                      ║
║  🤖 ML Classifier        →  Trained on labeled news data     ║
║       ↓                                                      ║
║  🏆 Verdict              →  ✅ REAL   or   ❌ FAKE           ║
╚══════════════════════════════════════════════════════════════╝
```

---

## ⚡ Tech Stack

```python
import pandas as pd                    # Load & manipulate News.csv
import seaborn as sns                  # Visualize class distributions
import matplotlib.pyplot as plt        # Plot confusion matrices & charts
# NLP Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer  # Text → numbers
from sklearn.model_selection import train_test_split          # Data splitting
from sklearn.pipeline import Pipeline                         # Clean ML workflow
# Classifiers
from sklearn.linear_model import LogisticRegression           # Baseline model
from sklearn.naive_bayes import MultinomialNB                 # Fast NLP model
from sklearn.ensemble import RandomForestClassifier           # Ensemble power
# Evaluation
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
```

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/fake-news-detection.git
cd fake-news-detection

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Fire up the notebook
jupyter lab Fake_news_detection.ipynb
```

---

## 📁 Project Structure

```
📦 fake-news-detection/
 ┣ 📓 Fake_news_detection.ipynb   ← Full NLP + ML pipeline (START HERE)
 ┣ 📊 News.csv                    ← Real & fake news article dataset
 ┣ 📄 requirements.txt            ← All dependencies
 ┗ 📖 README.md                   ← You're reading it
```

---

## 🔬 Step-by-Step Pipeline

```
STEP 1 ▸ Load Data          →  pd.read_csv('News.csv', index_col=0)
STEP 2 ▸ Explore (EDA)      →  .head(), .info(), class distribution plots
STEP 3 ▸ Preprocess Text    →  Combine title + text, clean & lowercase
STEP 4 ▸ Vectorize          →  TF-IDF transforms words into ML-ready features
STEP 5 ▸ Split Dataset      →  80% train / 20% test
STEP 6 ▸ Train Classifier   →  Fit model on vectorized news text
STEP 7 ▸ Predict            →  Model labels unseen articles REAL or FAKE
STEP 8 ▸ Evaluate           →  Accuracy, Classification Report, Confusion Matrix
```

---

## 📈 What TF-IDF Actually Does

> **TF-IDF (Term Frequency–Inverse Document Frequency)** is the brain of this system.
> It figures out which words *uniquely* define fake vs real articles — not just which words appear *most*, but which words *matter most*.

| Word Type | TF-IDF Score | Why |
|-----------|-------------|-----|
| Common words ("the", "is") | 🔻 Low | Appear everywhere — useless |
| Rare but meaningful words | 🔺 High | Unique signal → powerful feature |
| Sensationalist vocabulary | 🔺 High | Often flags fake news patterns |

---

## 💡 Key Takeaways

- 📰 **NLP is the backbone** of modern misinformation detection — this project proves it
- 🔢 **TF-IDF** transforms raw text into features that ML models can actually learn from
- ⚖️ **Class balance matters** — always check your label distribution before training
- 📊 **Confusion matrix** reveals what the model gets wrong — as important as accuracy

---

<div align="center">

---

## 🌟 Found this useful? Drop a ⭐ and share it!

![NLP](https://img.shields.io/badge/Domain-NLP%20%26%20Text%20Classification-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Fake News](https://img.shields.io/badge/Fight-Misinformation-FF4500?style=for-the-badge)
![Open Source](https://img.shields.io/badge/Open-Source-32CD32?style=for-the-badge&logo=github&logoColor=white)

---

### 🔗 Connect with the Author

[![GitHub](https://img.shields.io/badge/GitHub-Edmund%20Eric%20Gah-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Eddiegah)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Edmund%20Eric%20Gah-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/https://www.linkedin.com/in/edmund-eric-gah-7432a7213/)

---

<br/>

**Engineered with precision by**

### Edmund Eric Gah
*Machine Learning Engineer · NLP Practitioner · AI for Social Good*

> *"Technology is most powerful when it's used to protect people — not mislead them."*

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=2,12,24&height=100&section=footer"/>

</div>
