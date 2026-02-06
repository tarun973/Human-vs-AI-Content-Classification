# Human vs AI-Generated Text Classification (Deep Learning)

This project explores whether machine learning and deep learning models can reliably distinguish **human-written text** from **AI-generated text**.  
Rather than assuming more complex models are always better, this work systematically evaluates **statistical, sequential, transformer-based, and hybrid ensemble approaches** to understand *which signals truly matter* for this task.

---

## 🧩 Motivation
With the rapid adoption of large language models (LLMs), AI-generated text is now prevalent across:
- Academic submissions
- Online forums
- Journalism and misinformation pipelines

This raises concerns around **authenticity, integrity, and trust**.  
The central question of this project is:

> *Do deep contextual models actually outperform simpler lexical methods for detecting AI-generated text, or are stylistic cues sufficient?*

---

## 📊 Dataset
- **Source:** Hugging Face – *AI Text Detection Pile*
- **Total Samples:** 80,000  
  - 40,000 human-written  
  - 40,000 AI-generated
- **Labels:**
  - `0` → Human-written
  - `1` → AI-generated
- **Class Balance:** Perfectly balanced

A subset of the dataset was used due to local hardware constraints while preserving distributional balance.

---

## 🧼 Preprocessing Pipeline
A shared preprocessing pipeline was applied across models:
- Lowercasing
- URL, mention, and hashtag removal
- Removal of non-alphanumeric characters
- Stopword removal
- Tokenization and lemmatization

This ensured fair comparison across architectures.

---

## 🔍 Exploratory Data Analysis (EDA)
EDA revealed **clear stylistic differences** between classes:
- **Text Length:** Human-written text exhibits significantly higher variance and longer tails
- **Vocabulary Usage:** Human text shows richer lexical diversity; AI text favors repetitive, instructional phrasing
- **Punctuation Patterns:** Human text uses punctuation more variably and expressively

These observations hinted that **lexical and structural cues** may be highly predictive.

---

## 🧠 Models Evaluated

### 1️⃣ Baseline — TF-IDF + MLP
- TF-IDF vectorization (5,000 features)
- Multilayer Perceptron classifier
- Captures lexical and stylistic patterns

> **Purpose:** Establish whether simple word-distribution features are sufficient.

---

### 2️⃣ Sequential Models (GloVe 300d)
- **GRU (Gated Recurrent Unit)**
- **TCN (Temporal Convolutional Network)**
- **XGBoost** trained on embedding representations

> **Purpose:** Capture temporal and sequential dependencies beyond word frequency.

---

### 3️⃣ Hybrid Ensemble (Stacking)
- GRU + XGBoost
- TCN + XGBoost
- Final predictions stacked using Logistic Regression

> **Purpose:** Combine neural sequence modeling with strong decision boundaries.

---

### 4️⃣ Transformer Models
- **DistilBERT (fine-tuned)**
- **RoBERTa-base (fine-tuned)**

Minimal preprocessing was applied to leverage contextual representations learned from large corpora.

---

## 📈 Evaluation Methodology
All models were evaluated using **5-fold cross-validation** with:
- Accuracy
- Precision
- Recall
- F1-score
- ROC–AUC

---

## 🏆 Results Summary

| Model                    | Accuracy | F1-Score |
|-------------------------|----------|----------|
| TF-IDF + MLP            | **0.99** | **0.99** |
| GRU (GloVe)             | 0.88     | 0.88     |
| TCN (GloVe)             | 0.84     | 0.84     |
| XGBoost (GloVe)         | 0.77     | 0.77     |
| GRU + XGBoost Ensemble  | 0.88     | 0.88     |
| RoBERTa-base            | 0.94     | 0.94     |
| DistilBERT              | 0.93     | 0.93     |

---

## 🧠 Key Insights
- **Model complexity does not guarantee better performance**
- Lexical and stylistic patterns dominate in this dataset
- Transformers capture context well but are unnecessary when simpler cues suffice
- Ensembles improve robustness but not always raw accuracy
- Matching **model choice to data characteristics** is critical

---

## ⚠️ Error Analysis
- TF-IDF struggles with factual, list-based content that resembles human writing
- Transformers fail on noisy or low-context text
- Misclassifications highlight limits of stylistic-based detection

---

## 🛠️ Tech Stack
- **Languages:** Python
- **ML / DL:** PyTorch, scikit-learn, XGBoost
- **NLP:** Hugging Face Transformers, NLTK
- **Embeddings:** TF-IDF, GloVe (300d)
- **Models:** GRU, TCN, DistilBERT, RoBERTa

---

