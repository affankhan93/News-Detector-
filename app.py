import re
import joblib 
import nltk 
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout='centered',
    initial_sidebar_state='expanded',
)

# load model & vectorizer store in .pkl file
@st.cache_resource
def load_artifacts():
    model = joblib.load("randomforest_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

@st.cache_resource
def load_nltk_resources():
    nltk.download('stopwords', quiet=True)
    nltk.download("wordnet", quiet=True)
    return set(stopwords.words('english')), WordNetLemmatizer()

model, vectorizer = load_artifacts()
stopwords, lemmatizer = load_nltk_resources()

# clean text 
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stopwords and len(w) > 2]

    return " ".join(words)


# Custom Css 
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .title-text {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #ff512f, #dd2476);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .subtitle-text {
        text-align: center;
        color: #9ca3af;
        font-size: 16px;
        margin-top: 4px;
        margin-bottom: 30px;
    }
    .stTextArea textarea {
        border-radius: 12px;
        border: 1px solid #374151;
        font-size: 15px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3em;
        font-size: 18px;
        font-weight: 700;
        background: linear-gradient(90deg, #ff512f, #dd2476);
        color: white;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0px 0px 15px rgba(221, 36, 118, 0.6);
    }
    .result-box-real {
        padding: 20px;
        border-radius: 14px;
        background: linear-gradient(135deg, #134e4a, #065f46);
        border: 1px solid #10b981;
        text-align: center;
        font-size: 22px;
        font-weight: 700;
        color: #d1fae5;
        margin-top: 20px;
    }
    .result-box-fake {
        padding: 20px;
        border-radius: 14px;
        background: linear-gradient(135deg, #4c0519, #881337);
        border: 1px solid #ef4444;
        text-align: center;
        font-size: 22px;
        font-weight: 700;
        color: #fecaca;
        margin-top: 20px;
    }
    .confidence-text {
        text-align: center;
        color: #9ca3af;
        font-size: 15px;
        margin-top: 8px;
    }
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# header of the page
st.markdown('<p class="title-text">📰 Fake News Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text"> Paste a your news headline or article below to check its authenticity using a Machine Learning Model </p>', unsafe_allow_html=True)

# sidbar of the page
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        "This app uses a **Random Forest Classifier** trained on TF-IDF"
        " features from thousands of real and fake news articles to predict"
        " wheather a given news text is **Real** or **Fake**."
    )
    st.markdown('---')
    st.header("⚙️ How it works")
    st.write(
        "1. Enter or paste news text\n"
        "2. Text is cleaned & vectorized (TF-IDF)\n"
        "3. Model predicts Real / Fake\n"
        "4. Confidence score is displayed"
    )
    st.markdown("---")
    st.caption("Built with Streamlit • scikit-learn • NLTK")


# input area 
news_text = st.text_area(
    "Enter news headline / article text:", 
    height=200, placeholder="Paste the news article or headline here...."
)

col1, col2 = st.columns([1,1])
with col1:
    predict_button = st.button("🔍 Analyze News")   
with col2:
    clear_button = st.button("🗑️ Clear") 

if clear_button:
    st.rerun()

# prediction logic
if predict_button:
    if not news_text.strip():
        st.warning("Please enter some news text before analyzing.")
    else:
        with st.spinner("Analyzing Text..."):
            cleaned = clean_text(news_text)
            vec = vectorizer.transform([cleaned])
            prediction = model.predict(vec)[0]
            probabilities = model.predict_proba(vec)[0]
            confidence = max(probabilities) * 100

        if prediction == 1:
            st.markdown(
                f'<div class="result-box-real">✅ This News appears to be **REAL**</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-box-fake">❌ This News appears to be **FAKE**</div>',
                unsafe_allow_html=True
            )
        st.markdown(
            f'<p class="confidence-text"> Model Confidence: {confidence:.2f}%</p>',
            unsafe_allow_html=True
        )

        with st.spinner("View prediction probablities"):
            st.write(f"**Fake probability:** {probabilities[0]*100:.2f}%")
            st.write(f"**Real probability:** {probabilities[1]*100:.2f}%")
            st.progress(probabilities[1])


# Footer Note
st.markdown("---")
st.caption(
    "⚠️ Disclaimer: This is a *Academic* machine learning project trained on historical dataset."
    " Predictions should not be used as the sole basis for judging real-world news authenticity."
    )
