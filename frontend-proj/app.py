# app.py file
import streamlit as st

# Set page config
st.set_page_config(page_title="Hotel Feedback Dashboard", layout="wide")

# Title and navigation
st.title("🏨 Hotel Customer Feedback Dashboard")

st.sidebar.title("🔍 Navigation")
pages = {
    "1. Overall Customer Satisfaction & Sentiment": "pages/1_Overall_Sentiment.py",
}

selection = st.sidebar.radio("Go to", list(pages.keys()))

# Load selected page
with open(pages[selection], encoding="utf-8") as f:
    code = compile(f.read(), pages[selection], 'exec')
    exec(code)