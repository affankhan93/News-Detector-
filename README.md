# News-Detector-
(This is a Academic Project)

## 📰 Fake News Detector
A supervised machine learning web application that classifies news articles or headlines as Real or Fake, built as an end-to-end ML project — from raw data to a deployed, interactive UI.

## 🔍 Overview
This project uses TF-IDF vectorization combined with classical supervised learning algorithms (Logistic Regression, Multinomial Naive Bayes, and Random Forest) to classify news text as real or fake. The final model is deployed through a Streamlit web app with a custom, polished UI, allowing users to paste any news headline or article and instantly get a prediction with a confidence score.

## 📊 Dataset
- Source: [Fake and Real News Dataset (ISOT)](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?select=Fake.csv)
- Files used: Fake.csv, True.csv
- Size: ~44,000 labeled news articles combined
- Columns: title, text, subject, date
- Label: 0 = Fake, 1 = Real

## 🧠 Approach / Pipeline
1. **Data Loading & Labeling** — Combined Fake.csv and True.csv, added binary labels, shuffled the dataset.
2. **Exploratory Data Analysis (EDA)** — Checked class balance, text length distribution, and identified the Reuters leakage pattern.
3. **Leakage Mitigation** — Removed Reuters datelines/mentions from article text using regex.
4. **Text Preprocessing** — Lowercasing, URL removal, punctuation/number stripping, stopword removal, and lemmatization (NLTK).
5. **Train/Test Split** — 80/20 split performed before vectorization to prevent data leakage from the test set into feature extraction.
6. **Feature Extraction** — TF-IDF vectorization (max_features=5000, English stopwords).
7. **Model Training** — Trained and compared three supervised models:
- *Logistic Regression*
- *Multinomial Naive Bayes*
- *Random Forest Classifier*
8. **Evaluation** — Accuracy, Precision, Recall, F1-score, and Confusion Matrix for each model.
9. **Model Persistence** — Saved the best-performing model and the fitted TF-IDF vectorizer using joblib.
10. **Deployment** — Built an interactive Streamlit web app for real-time predictions.

## 🏆 Model Performance

| Model                     | Accuracy | Precision | 
|----------------------------|----------|-----------|
| Logistic Regression        | *97.98%*    | *97.28%*     | 
| Multinomial Naive Bayes    | *92.71%*    | *93.05%*     | 
| Random Forest Classifier   | *98.38%*    | *97.93%*     | 

**Final model used in deployment**: Random Forest Classifier

## 🛠️ Tech Stack

| Category                     | Tools / Libraries                                              |
|-------------------------------|------------------------------------------------------------------|
| Language                      | Python 3.13                                                      |
| Data Handling                  | pandas, numpy                                                    |
| NLP Preprocessing              | NLTK (stopwords, WordNetLemmatizer), re                          |
| Feature Extraction             | scikit-learn (TfidfVectorizer)                                   |
| ML Models                      | scikit-learn (LogisticRegression, MultinomialNB, RandomForestClassifier) |
| Model Persistence              | joblib                                                            |
| Web App / Deployment           | Streamlit                                                         |
| Environment / Package Manager  | uv, venv
|

## 📁 Project Structure

```
Fake_News_Detector/
├── app.py                      # Streamlit web application
├── Fake_News_detector.ipynb    # Full notebook: EDA, preprocessing, training, evaluation
├── requirements.txt            # Project dependencies
├── randomforest_model.pkl      # Trained Random Forest model
├── tfidf_vectorizer.pkl        # Fitted TF-IDF vectorizer
├── Fake.csv                    # Fake news dataset (not included in repo — see Dataset section)
├── True.csv                    # Real news dataset (not included in repo — see Dataset section)
└── README.md                   # Project documentation
```

## 🚀 Usage
1. Paste a news headline or full article text into the input box.
2. Click Analyze News.
3. The app displays:
- A Real or Fake prediction
- A model confidence score
- A probability breakdown for both classes

## 📌 Key Learnings & Highlights
- Practiced the full supervised ML lifecycle — from raw text to deployed application.
- Identified and addressed data leakage, a critical but often-overlooked issue in ML pipelines, demonstrating rigorous model validation rather than blindly trusting high accuracy scores.
- Applied NLP preprocessing techniques (tokenization, lemmatization, stopword removal) suited for text classification tasks.
- Compared multiple classical ML algorithms and evaluated trade-offs between them for text data.
- Built and deployed a user-facing product, not just a notebook — reinforcing skills in application development and deployment, not only modeling.

## ⚠️ Disclaimer
This project is built for educational and portfolio purposes using a historical, static dataset. It should *not* be used as a sole or authoritative tool for verifying real-world news authenticity. Misinformation detection in production systems requires more robust, continuously updated, and multi-source verification approaches.
