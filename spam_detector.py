import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

# Ensure required NLTK data is available
nltk.download('stopwords')
nltk.download('wordnet')

# UI Styling
st.markdown("""
<style>
    
</style>
""", unsafe_allow_html=True)

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    words = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in words])

# Load vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# UI Title
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
