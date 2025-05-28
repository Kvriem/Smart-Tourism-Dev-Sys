import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter
import ast

st.set_page_config(page_title="Demographic Insights", layout="wide")

st.title("👥 Demographic Insights")


# --- Helper functions ---
def safe_parse_list(x):
    try:
        return ast.literal_eval(x)
    except:
        return []


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



@st.cache_data

def load_data():
    df = pd.read_csv('Pasted_Text_1748194864747.txt', on_bad_lines='skip', encoding='utf-8')
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace(',', '')
    df['Sentiment_Label'] = df['sentiment_classification'].map({-1: 'Negative', 1: 'Positive'})
    return df

df = load_data()

st.markdown("### 📊 Key Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    top_nat = df['Reviewer_Nationality'].value_counts().idxmax()
    top_nat_count = df['Reviewer_Nationality'].value_counts().max()
    st.metric("Top Reviewer Nationality", top_nat, f"{top_nat_count} reviews")

with col2:
  # --- Visualizations ---
    st.markdown("### 📈 Visualizations")
    top_travel_type = df['Travel_Type'].value_counts().idxmax()
    st.metric("Top Travel Type", top_travel_type)

with col3:
    top_room = df['Room_Type'].value_counts().idxmax()
    st.metric("Most Booked Room Type", top_room)


gap()
# Pie Charts
col4, col5 = st.columns(2)

with col4:
    nat_counts = df['Reviewer_Nationality'].value_counts().head(10)
    fig = px.pie(names=nat_counts.index, values=nat_counts.values, title="🌍 Top Reviewer Nationalities")
    st.plotly_chart(fig, use_container_width=True)

with col5:
    travel_counts = df['Travel_Type'].value_counts()
    fig = px.pie(names=travel_counts.index, values=travel_counts.values, title="🧳 Travel Type Breakdown")
    st.plotly_chart(fig, use_container_width=True)

# Stacked Bar: Sentiment by Travel Type
st.subheader("📊 Sentiment by Travel Type")
sentiment_by_trip = df.groupby(['Travel_Type', 'Sentiment_Label']).size().reset_index(name='Count')
fig = px.bar(sentiment_by_trip, x='Travel_Type', y='Count', color='Sentiment_Label', barmode='stack',
             title="Stacked Sentiment Distribution by Travel Type", color_discrete_sequence=px.colors.qualitative.Set2)
st.plotly_chart(fig, use_container_width=True)

# Bubble Chart: Room Type Popularity vs Satisfaction
st.subheader("🟢 Room Type Popularity vs Satisfaction")
room_stats = df.groupby('Room_Type').agg({
    'sentiment_classification': 'mean',
    'Reviewer_Nationality': 'count'
}).rename(columns={'Reviewer_Nationality': 'Review_Count', 'sentiment_classification': 'Avg_Sentiment'}).reset_index()
fig = px.scatter(room_stats, x='Review_Count', y='Avg_Sentiment', size='Review_Count', color='Avg_Sentiment',
                 hover_name='Room_Type', color_continuous_scale='Viridis',
                 title="Room Type Popularity vs Avg Sentiment")
st.plotly_chart(fig, use_container_width=True)
