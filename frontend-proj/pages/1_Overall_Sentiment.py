import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter
import ast  # For safe parsing of stringified lists

# Custom CSS
st.markdown("""
    <style>
        body {
            font-family: 'Segoe UI', sans-serif;
            background-color: #f4f6f8;
            color: #333;
        }
        h2, h3 {
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

# Helper function to safely parse lists from strings
def safe_parse_list(x):
    try:
        return ast.literal_eval(x)
    except:
        return []

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('data/Pasted_Text_1748194864747.txt', on_bad_lines='skip', encoding='utf-8')
    df.columns = df.columns.str.strip().str.replace(' ', '_', regex=True).str.replace(',', '', regex=True)
    df['Sentiment_Label'] = df['sentiment_classification'].map({-1: 'Negative', 1: 'Positive'})
    return df

df = load_data()

# List of cities from the dataset
egypt_cities = sorted(df['City'].dropna().unique())
city_options = ["Overall"] + list(egypt_cities)

# City Selector
selected_city = st.selectbox("🏙️ Select a city", city_options)

# Filter data based on city
if selected_city != "Overall":
    filtered_df = df[df['City'] == selected_city]
else:
    filtered_df = df

# Check if filtered_df has any data
if len(filtered_df) == 0:
    st.warning("⚠️ No data found for the selected city.")
    st.stop()

# --- METRICS ROW ---
st.markdown("### 📊 Key Metrics")
col1, col2, col3 = st.columns(3)

# Metric 1: Total Reviews
with col1:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.markdown("#### Total Reviews")
    st.markdown(f"<h3 style='text-align:center;'>📝 {len(filtered_df):,}</h3>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Metric 2: Guest Satisfaction
with col2:
    avg_sentiment = filtered_df['sentiment_classification'].mean()
    if avg_sentiment > 0.6:
        sentiment_label = "😊 Very Positive"
        color = "#4CAF50"
    elif avg_sentiment > 0.2:
        sentiment_label = "🙂 Positive"
        color = "#8BC34A"
    elif avg_sentiment > -0.2:
        sentiment_label = "😐 Neutral"
        color = "#FF9800"
    elif avg_sentiment > -0.6:
        sentiment_label = "🙁 Negative"
        color = "#F44336"
    else:
        sentiment_label = "😟 Very Negative"
        color = "#D32F2F"
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.markdown("#### 😊 Overall Guest Satisfaction")
    st.markdown(f"<h3 style='text-align:center; color:{color};'>{sentiment_label}</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center; font-size:0.9em; color:#666;'>Based on {len(filtered_df)} reviews</p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Metric 3: Positive Reviews Count
with col3:
    positive_count = len(filtered_df[filtered_df['Sentiment_Label'] == 'Positive'])
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.markdown("#### 👍 Positive Reviews")
    st.markdown(f"<h3 style='text-align:center;'>{positive_count}</h3>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- CHARTS IN GRID LAYOUT ---
st.markdown("### 📊 Dashboard Overview")

# Full Width Row: Top Reviewed Hotels
st.markdown("#### 🏨 Top Reviewed Hotels")
top_hotels = filtered_df['Hotel_Name'].value_counts().head(10).sort_values()
fig = px.bar(
    x=top_hotels.values,
    y=top_hotels.index,
    orientation='h',
    color=top_hotels.values,
    color_continuous_scale='Blues'
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("🔍 View Detailed Hotel Sentiment"):
    hotel_sentiment = filtered_df.groupby('Hotel_Name')['sentiment_classification'].mean().sort_values(ascending=False)
    if not hotel_sentiment.empty:
        fig_detail = px.bar(
            hotel_sentiment,
            x=hotel_sentiment.values,
            y=hotel_sentiment.index,
            orientation='h',
            color=hotel_sentiment.values,
            color_continuous_scale='RdBu',
            title="📊 Avg Sentiment per Hotel",
            labels={'x': 'Avg Sentiment Score', 'y': ''}
        )
        st.plotly_chart(fig_detail, use_container_width=True)
    else:
        st.info("No hotel sentiment data available.")

# Two Charts Side-by-Side
row1_col1, row1_col2 = st.columns([2, 2])

# Pie Chart: Positive vs Negative
with row1_col1:
    st.markdown("#### 📈 % Positive vs Negative Reviews")
    labels = ['Positive', 'Negative']
    sizes = [len(filtered_df[filtered_df['Sentiment_Label'] == 'Positive']),
             len(filtered_df[filtered_df['Sentiment_Label'] == 'Negative'])]
    fig = px.pie(names=labels, values=sizes, color_discrete_sequence=px.colors.qualitative.Prism)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("🔍 View Nationality Breakdown"):
        top_nationalities = filtered_df['Reviewer_Nationality'].value_counts().head(10)
        if not top_nationalities.empty:
            fig_detail = px.pie(
                names=top_nationalities.index,
                values=top_nationalities.values,
                title="🌍 Reviewer Nationality Breakdown",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            st.plotly_chart(fig_detail, use_container_width=True)
        else:
            st.info("No nationality data available.")

# Bar Chart: Top Reviewer Nationalities
with row1_col2:
    st.markdown("#### 🌎 Top Reviewer Nationalities")
    top_nationalities = filtered_df['Reviewer_Nationality'].value_counts().head(10).sort_values()
    fig = px.bar(
        x=top_nationalities.values,
        y=top_nationalities.index,
        orientation='h',
        color=top_nationalities.values,
        color_continuous_scale='Oranges'
    )
    st.plotly_chart(fig, use_container_width=True)

# Word Frequency Charts
row3_col1, row3_col2 = st.columns([2, 2])

# Top 10 Positive Words
with row3_col1:
    st.markdown("#### 🔝 Top 10 Positive Words")
    
    # Safely extract positive words
    pos_tokens = filtered_df[filtered_df['Sentiment_Label'] == 'Positive']['positive_tokens'].apply(safe_parse_list)
    pos_words = [word.strip() for sublist in pos_tokens for word in sublist if isinstance(word, str) and word.strip()]
    
    if pos_words:
        word_counts = Counter(pos_words)
        top_words = pd.DataFrame(word_counts.most_common(10), columns=['Word', 'Count'])
        fig = px.bar(
            top_words,
            x='Count',
            y='Word',
            orientation='h',
            color='Count',
            color_continuous_scale='Greens'
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No positive words found.")

    with st.expander("🔍 View Word Frequency Distribution"):
        if pos_words:
            fig_detail = px.scatter(
                top_words,
                x='Word',
                y='Count',
                size='Count',
                color='Count',
                color_continuous_scale='Viridis',
                title="🧮 Positive Words Frequency Distribution",
                height=800
            )
            st.plotly_chart(fig_detail, use_container_width=True)
        else:
            st.info("No data to display.")

# Top 10 Negative Words
with row3_col2:
    st.markdown("#### ⚠️ Top 10 Negative Words")
    
    neg_tokens = filtered_df[filtered_df['Sentiment_Label'] == 'Negative']['negative_tokens'].apply(safe_parse_list)
    neg_words = [word.strip() for sublist in neg_tokens for word in sublist if isinstance(word, str) and word.strip()]
    
    if neg_words:
        word_counts = Counter(neg_words)
        top_words_neg = pd.DataFrame(word_counts.most_common(10), columns=['Word', 'Count'])
        fig = px.bar(
            top_words_neg,
            x='Count',
            y='Word',
            orientation='h',
            color='Count',
            color_continuous_scale='Reds'
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No negative words found.")

    with st.expander("🔍 View Negative Word Distribution"):
        if neg_words:
            fig_detail = px.scatter(
                top_words_neg,
                x='Word',
                y='Count',
                size='Count',
                color='Count',
                color_continuous_scale='Inferno',
                title="🟥 Negative Words Frequency Distribution",
                height=800
            )
            st.plotly_chart(fig_detail, use_container_width=True)
        else:
            st.info("No data to display.")

# City-wise Sentiment Comparison (Only show when "Overall" is selected)
if selected_city == "Overall":
    st.markdown("#### 🗺️ City-wise Sentiment Comparison")
    city_sentiment = df.groupby('City')['sentiment_classification'].mean().reset_index()
    fig = px.bar(
        city_sentiment,
        x='sentiment_classification',
        y='City',
        orientation='h',
        color='sentiment_classification',
        color_continuous_scale='RdYlGn',
        labels={'sentiment_classification': 'Avg Sentiment Score', 'City': ''},
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("🔍 View City Sentiment Details"):
        st.markdown("#### 📊 Distribution of Sentiment by City")
        city_sentiment_hist = df.groupby('City')['sentiment_classification'].apply(list).reset_index()
        if not city_sentiment_hist.empty:
            fig_detail = px.box(
                city_sentiment_hist,
                x='City',
                y='sentiment_classification',
                color='City',
                title="📊 Distribution of Sentiment by City",
                color_discrete_sequence=px.colors.qualitative.Set2,
                height=800
            )
            st.plotly_chart(fig_detail, use_container_width=True)
        else:
            st.info("No city sentiment details available.")