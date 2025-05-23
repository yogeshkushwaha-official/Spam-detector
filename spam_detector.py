import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# App title with style
st.markdown("""
    <h1 style='text-align: center; color: #00FFAA; font-size: 3em;'>📨 Spam Message Detector</h1>
    <p style='text-align: center; color: #BBBBBB;'>Check if a message is spam or safe in real-time!</p>
    <hr style='border: 1px solid #444;'/>
""", unsafe_allow_html=True)

# Custom styling
st.markdown("""
<style>
    .stApp {
        background-color: #0f1117;
        color: white;
    }
    .stTextArea > div > textarea {
        background-color: #1e1e1e;
        color: white;
        border-radius: 0.5rem;
        border: 1px solid #444;
        font-size: 16px;
    }
    .stButton > button {
        background-color: #00FFAA;
        color: black;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1.5rem;
        font-weight: bold;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Preprocessing
def transform_text(text):
    text = text.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    words = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in words])

# Message input
st.markdown("<label style='color:white; font-weight:bold; font-size:16px;'>📩 Enter your message below:</label>", unsafe_allow_html=True)
message = st.text_area(label="", key="text_input")

# Predict button
if st.button("🔍 Predict"):
    if message.strip():
        transformed_sms = transform_text(message)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        proba = model.predict_proba(vector_input)[0]
        confidence = round(max(proba) * 100, 2)

        if result == 1:
            st.markdown(f"""
            <div style='text-align: center; padding: 1rem; background-color: #330000; border-radius: 0.5rem;'>
                <h2 style='color: #FF5555;'>🚨 This message is <strong>SPAM</strong></h2>
                <p>⚠️ Confidence: <strong>{confidence}%</strong></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='text-align: center; padding: 1rem; background-color: #003300; border-radius: 0.5rem;'>
                <h2 style='color: #00FF88;'>✅ This message is <strong>NOT SPAM</strong></h2>
                <p>👍 Confidence: <strong>{confidence}%</strong></p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a message before predicting.")
