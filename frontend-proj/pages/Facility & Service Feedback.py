import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import ast

st.set_page_config(page_title="Facility & Service Feedback", layout="wide")

st.title("🛎️ Facility & Service Feedback")


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
    df['Sentiment_Label'] = df['sentiment_classification'].map({-1: 'Negative', 1: 'Positive'})
    return df

df = load_data()

# --- Key Metrics ---
st.markdown("### 📊 Key Metrics")
facilities = ['breakfast', 'pool', 'staff', 'cleanliness', 'location', 'wifi', 'bed', 'noise', 'room', 'reception']

flat_pos = [w for sub in df['positive_tokens'].apply(safe_parse_list) for w in sub if isinstance(w, str)]
flat_neg = [w for sub in df['negative_tokens'].apply(safe_parse_list) for w in sub if isinstance(w, str)]

pos_facility = Counter([w for w in flat_pos if w in facilities])
neg_facility = Counter([w for w in flat_neg if w in facilities])

most_praised = pos_facility.most_common(1)[0] if pos_facility else ("None", 0)
most_faulted = neg_facility.most_common(1)[0] if neg_facility else ("None", 0)

accessibility_terms = ['access', 'wheelchair', 'stairs', 'elevator', 'comfort']
access_issues = sum([flat_neg.count(term) for term in accessibility_terms])

col1, col2, col3 = st.columns(3)
col1.metric("Most Praised Facility", most_praised[0], most_praised[1])
col2.metric("Most Faulted Facility", most_faulted[0], most_faulted[1])
col3.metric("Accessibility Issues Mentioned", "Issues", access_issues)

# --- Tag Cloud ---
st.markdown("### ☁️ Facility Mention Cloud")
facility_counts = Counter([w for w in flat_pos + flat_neg if w.lower() in facilities])
facility_df = pd.DataFrame(facility_counts.items(), columns=['Facility', 'Count']).sort_values(by='Count', ascending=False)
fig = px.treemap(facility_df, path=['Facility'], values='Count', title="Facility Mentions in Reviews")
st.plotly_chart(fig, use_container_width=True)

# --- Radar Chart ---
st.markdown("### 📍 Facility Sentiment Radar")
radar_data = []
for facility in facilities:
    pos = df['positive_tokens'].apply(lambda tokens: facility in safe_parse_list(tokens)).sum()
    neg = df['negative_tokens'].apply(lambda tokens: facility in safe_parse_list(tokens)).sum()
    total = pos + neg
    score = (pos - neg) / total if total else 0
    radar_data.append((facility, score))

radar_df = pd.DataFrame(radar_data, columns=['Facility', 'Sentiment_Score'])
fig = go.Figure()
fig.add_trace(go.Scatterpolar(r=radar_df['Sentiment_Score'], theta=radar_df['Facility'], fill='toself', name='Facility Scores'))
fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[-1, 1])), showlegend=False, title="Facility Sentiment Radar Chart")
st.plotly_chart(fig, use_container_width=True)

# --- Tables: Top Complaints ---
st.markdown("### 🔴 Common Complaints per Facility")
complaints = []
for facility in facilities:
    related = df[df['negative_tokens'].apply(lambda tokens: facility in safe_parse_list(tokens))]
    for _, row in related.iterrows():
        complaints.append((facility, row['Negative_Review'], row['Hotel_Name']))
complaints_df = pd.DataFrame(complaints, columns=['Facility', 'Complaint', 'Hotel_Name'])
st.dataframe(complaints_df.head(50), use_container_width=True)

# --- Word Clouds on Staff/Service ---
st.markdown("### 👨‍💼 Feedback on Staff & Service")
filtered_staff = df[df['positive_tokens'].apply(lambda x: 'staff' in safe_parse_list(x)) | df['negative_tokens'].apply(lambda x: 'staff' in safe_parse_list(x))]

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### ⭐ Praise About Staff")
    pos_words = filtered_staff[filtered_staff['Sentiment_Label'] == 'Positive']['positive_tokens'].apply(safe_parse_list)
    pos_flat = [w for sub in pos_words for w in sub if isinstance(w, str)]
    pos_count = Counter(pos_flat).most_common(20)
    df_pos = pd.DataFrame(pos_count, columns=['Word', 'Count'])
    fig = px.bar(df_pos, x='Count', y='Word', orientation='h', color='Count', color_continuous_scale='Greens')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("#### ⚠️ Complaints About Staff")
    neg_words = filtered_staff[filtered_staff['Sentiment_Label'] == 'Negative']['negative_tokens'].apply(safe_parse_list)
    neg_flat = [w for sub in neg_words for w in sub if isinstance(w, str)]
    neg_count = Counter(neg_flat).most_common(20)
    df_neg = pd.DataFrame(neg_count, columns=['Word', 'Count'])
    fig = px.bar(df_neg, x='Count', y='Word', orientation='h', color='Count', color_continuous_scale='Reds')
    st.plotly_chart(fig, use_container_width=True)
