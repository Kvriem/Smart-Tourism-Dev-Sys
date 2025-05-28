import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter
import ast

st.set_page_config(page_title="Hotel Performance & Comparison", layout="wide")



# Helper function for vertical spacing
def gap(size=2.5):
    """Adds vertical spacing/gap."""
    st.markdown(f"<div style='margin-top: {size}rem;'></div>", unsafe_allow_html=True)

# Custom CSS
st.markdown("""
    <style>
        body {
            font-family: 'Segoe UI', sans-serif;
            background-color: #f4f6f8;
            color: #333;
        }
        h2, h3, h4 {
            color: #1f283e;
        }
        .metric-box {
            background-color: #ffffff;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🏨 By Hotel Performance & Comparison")


# --- Helper functions ---
def safe_parse_list(x):
    try:
        return ast.literal_eval(x)
    except:
        return []

@st.cache_data

def load_data():
    df = pd.read_csv('Pasted_Text_1748194864747.txt', on_bad_lines='skip', encoding='utf-8')
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace(',', '')
    df['Sentiment_Label'] = df['sentiment_classification'].map({-1: 'Negative', 1: 'Positive'})
    return df

df = load_data()

# --- Key Metrics ---
st.markdown("### 📊 Key Metrics")

reviews_per_hotel = df['Hotel_Name'].value_counts()
avg_sentiment_per_hotel = df.groupby('Hotel_Name')['sentiment_classification'].mean()

most_reviewed_hotel = reviews_per_hotel.idxmax()
most_reviewed_count = reviews_per_hotel.max()

top_rated_hotel = avg_sentiment_per_hotel.idxmax()
top_sentiment_score = avg_sentiment_per_hotel.max()

unique_hotels = df['Hotel_Name'].nunique()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Most Reviewed Hotel", most_reviewed_hotel, f"{most_reviewed_count} reviews")
with col2:
    st.markdown("### 📈 Visualizations")
    st.metric("Top Rated Hotel", top_rated_hotel, f"{top_sentiment_score:.2f} sentiment")
with col3:
    st.markdown("### 🏨 Hotel count")
    st.metric("Total Hotels", unique_hotels)

# --- Visualizations ---

gap()
# Bar chart: Hotels ranked by sentiment score
st.subheader("📊 Hotels Ranked by Sentiment Score")
hotel_sentiment = avg_sentiment_per_hotel.sort_values(ascending=False).head(20)
fig = px.bar(
    x=hotel_sentiment.values,
    y=hotel_sentiment.index,
    orientation='h',
    labels={'x': 'Average Sentiment', 'y': 'Hotel'},
    color=hotel_sentiment.values,
    color_continuous_scale='RdYlGn'
)
st.plotly_chart(fig, use_container_width=True)

# Table: Top feedback per hotel
st.subheader("📝 Top Feedback per Hotel")
top_hotels = df['Hotel_Name'].value_counts().head(5).index
feedback = df[df['Hotel_Name'].isin(top_hotels)][['Hotel_Name', 'Positive_Review', 'Negative_Review', 'sentiment_classification']]
st.dataframe(feedback.head(100), use_container_width=True)

# Bubble Chart: Reviews vs sentiment
st.subheader("🟢 Bubble Chart: Reviews vs Sentiment")
hotel_stats = df.groupby('Hotel_Name').agg({
    'Positive_Review': 'count',
    'sentiment_classification': 'mean'
}).rename(columns={'Positive_Review': 'Review_Count', 'sentiment_classification': 'Avg_Sentiment'})
hotel_stats = hotel_stats[hotel_stats['Review_Count'] > 10].sort_values(by='Review_Count', ascending=False).head(30)
hotel_stats.reset_index(inplace=True)
fig_bubble = px.scatter(
    hotel_stats,
    x='Review_Count',
    y='Avg_Sentiment',
    size='Review_Count',
    color='Avg_Sentiment',
    hover_name='Hotel_Name',
    color_continuous_scale='Turbo',
    title="Reviews vs Average Sentiment"
)
st.plotly_chart(fig_bubble, use_container_width=True)

# Word clouds: praise/complaints
st.subheader("☁️ Word Clouds per Hotel")
hotel_choice = st.selectbox("Select a hotel", sorted(df['Hotel_Name'].dropna().unique()))

col3, col4 = st.columns(2)

with col3:
    st.markdown("#### ⭐ Most Common Praise Words")
    pos_tokens = df[(df['Hotel_Name'] == hotel_choice) & (df['Sentiment_Label'] == 'Positive')]['positive_tokens'].apply(safe_parse_list)
    pos_words = [w for sub in pos_tokens for w in sub if isinstance(w, str)]
    pos_counts = Counter(pos_words).most_common(20)
    pos_df = pd.DataFrame(pos_counts, columns=['Word', 'Count'])
    fig = px.bar(pos_df, x='Count', y='Word', orientation='h', color='Count', color_continuous_scale='Greens')
    st.plotly_chart(fig, use_container_width=True)

with col4:
    st.markdown("#### ⚠️ Most Common Complaint Words")
    neg_tokens = df[(df['Hotel_Name'] == hotel_choice) & (df['Sentiment_Label'] == 'Negative')]['negative_tokens'].apply(safe_parse_list)
    neg_words = [w for sub in neg_tokens for w in sub if isinstance(w, str)]
    neg_counts = Counter(neg_words).most_common(20)
    neg_df = pd.DataFrame(neg_counts, columns=['Word', 'Count'])
    fig = px.bar(neg_df, x='Count', y='Word', orientation='h', color='Count', color_continuous_scale='Reds')
    st.plotly_chart(fig, use_container_width=True)
