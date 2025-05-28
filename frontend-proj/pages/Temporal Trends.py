import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
import ast

st.set_page_config(page_title="Temporal Trends", layout="wide")

st.title("📅 Temporal Trends")


# --- Helper function ---
def safe_parse_list(x):
    try:
        return ast.literal_eval(x)
    except:
        return []

@st.cache_data

def load_data():
    df = pd.read_csv("Pasted_Text_1748194864747.txt", on_bad_lines='skip', encoding='utf-8')
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace(",", "")
    df['Review_Date'] = pd.to_datetime(df['Review_Date'], errors='coerce')
    df['Sentiment_Label'] = df['sentiment_classification'].map({-1: 'Negative', 1: 'Positive'})
    df['Month'] = df['Review_Date'].dt.to_period('M')
    df['Year'] = df['Review_Date'].dt.year
    df['Month_Name'] = df['Review_Date'].dt.strftime('%B')
    df['Season'] = df['Review_Date'].dt.month % 12 // 3
    df['Season'] = df['Season'].map({0: 'Winter', 1: 'Spring', 2: 'Summer', 3: 'Fall'})
    return df

df = load_data()

# --- Key Metrics ---
st.markdown("### 📊 Key Metrics")

monthly_counts = df.groupby('Month').size()
monthly_sentiment = df.groupby('Month')['sentiment_classification'].mean()

peak_month = monthly_counts.idxmax().strftime('%B %Y')
low_month = monthly_counts.idxmin().strftime('%B %Y')

col1, col2, col3 = st.columns(3)
col1.metric("Month with Most Reviews", peak_month, int(monthly_counts.max()))
col2.metric("Month with Least Reviews", low_month, int(monthly_counts.min()))
col3.metric("Avg Sentiment (All Time)", f"{df['sentiment_classification'].mean():.2f}")

# --- Line Chart: Sentiment Over Time ---
st.subheader("📈 Average Sentiment Over Time")
sentiment_time = df.groupby('Month')['sentiment_classification'].mean().reset_index()
sentiment_time['Month'] = sentiment_time['Month'].astype(str)
fig = px.line(sentiment_time, x='Month', y='sentiment_classification', markers=True, title="Sentiment Over Time")
st.plotly_chart(fig, use_container_width=True)

# --- Area Chart: Review Volume Over Months ---
st.subheader("📊 Monthly Review Volume")
review_volume = df.groupby('Month').size().reset_index(name='Review_Count')
review_volume['Month'] = review_volume['Month'].astype(str)
fig = px.area(review_volume, x='Month', y='Review_Count', title="Review Volume Over Time")
st.plotly_chart(fig, use_container_width=True)

# --- Calendar Heatmap (Fixed Aggregation) ---
st.subheader("🗓️ Calendar Heatmap of Reviews")
calendar_df = df.dropna(subset=['Review_Date'])
calendar_counts = calendar_df.groupby(calendar_df['Review_Date'].dt.date).size().reset_index(name='Review_Count')
calendar_counts['Date'] = pd.to_datetime(calendar_counts['Review_Date'])
calendar_counts = calendar_counts.set_index('Date')

fig, ax = plt.subplots(figsize=(14, 4))
ax.bar(calendar_counts.index, calendar_counts['Review_Count'], width=1, color='skyblue')
ax.set_title("Daily Review Volume")
ax.set_xlabel("Date")
ax.set_ylabel("Review Count")
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=45)
st.pyplot(fig)

# --- Scatter Plot: Sentiment vs Season ---
st.subheader("🌤️ Sentiment by Season")
season_sentiment = df.groupby(['Season'])['sentiment_classification'].mean().reset_index()
fig = px.scatter(season_sentiment, x='Season', y='sentiment_classification',
                 size=[30]*len(season_sentiment), color='Season', title="Sentiment by Season")
st.plotly_chart(fig, use_container_width=True)
