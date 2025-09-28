import streamlit as st
import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer

# ===============================
# Load Model and Tokenizer
# ===============================
from transformers import TFBertForSequenceClassification, BertTokenizer

MODEL_PATH = r"C:\Users\SONIC LAPTOPS\Desktop\all data\fake news -20250927T194318Z-1-001\model"

# Load model and tokenizer
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = TFBertForSequenceClassification.from_pretrained(MODEL_PATH)


# ===============================
# Page Configuration
# ===============================
st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="centered")

st.title("📰 Fake News Detection")
st.write("Enter a news article below to check if it's **Fake** or **Real**.")

# ===============================
# Prediction Function
# ===============================
def predict(text):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=256)
    outputs = model(inputs)
    logits = outputs.logits
    probs = tf.nn.softmax(logits, axis=1).numpy()[0]
    pred = tf.argmax(probs).numpy()
    label = "Fake" if pred == 1 else "Real"
    confidence = float(probs[pred]) * 100
    return label, confidence

# ===============================
# User Input
# ===============================
news_text = st.text_area("Paste your news article here:", height=200)

if st.button("Check News", use_container_width=True):
    if news_text.strip():
        label, confidence = predict(news_text)
        if label == "Real":
            st.success(f"✅ Prediction: {label} ({confidence:.2f}% confidence)")
        else:
            st.error(f"❌ Prediction: {label} ({confidence:.2f}% confidence)")
    else:
        st.warning("Please enter some text to analyze.")

# ===============================
# Example Section
# ===============================
st.markdown("---")
st.subheader("🔎 Try Example Articles")

examples = {
    "Real": "NASA confirms successful landing of Mars rover Perseverance in 2021.",
    "Fake": "Scientists confirm that drinking two liters of Coca-Cola daily cures diabetes."
}

choice = st.selectbox("Choose an example:", list(examples.keys()))

if st.button("Run Example"):
    example_text = examples[choice]
    st.write(example_text)
    label, confidence = predict(example_text)
    if label == "Real":
        st.success(f"✅ Prediction: {label} ({confidence:.2f}% confidence)")
    else:
        st.error(f"❌ Prediction: {label} ({confidence:.2f}% confidence)")
