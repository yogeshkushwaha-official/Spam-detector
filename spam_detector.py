import streamlit as st
import pickle
import string
import nltk
nltk.data.path.append('nltk_data')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
def setup_nltk():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

setup_nltk()

# Safely download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Set black background and white text
st.markdown(
    """
    <style>
    .stApp {
        background-color: #000000;
        color: white;
    }
    h1, h2, h3, h4, h5, h6, p, div, label {
        color: white !important;
    }
    .stTextInput > div > div > input,
    .stTextArea > div > textarea {
        background-color: #111 !important;
        color: white !important;
        border: 1px solid #444;
    }
    .stButton > button {
        background-color: #222;
        color: white;
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)




# Preprocessing function
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

def transform_text(text):
    text = text.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)  # Doesn't need punkt
    words = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in words])


# Load TF-IDF and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit UI
st.title("📩 Email/SMS Spam Classifier")

# Input box
message = st.text_area("Enter the message:")

# Predict button
if st.button("🔍 Predict") and message.strip():
    transformed_sms = transform_text(message)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    proba = model.predict_proba(vector_input)[0]
    confidence = round(max(proba) * 100, 2)

    if result == 1:
        st.markdown('<h2 style="color:red;">🚨 Spam</h2>', unsafe_allow_html=True)
    else:
        st.markdown('<h2 style="color:green;">✅ Not Spam</h2>', unsafe_allow_html=True)
    
    st.write(f"**Confidence:** {confidence}%")

