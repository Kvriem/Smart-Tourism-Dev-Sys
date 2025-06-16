import streamlit as st
import plotly.express as px
import pandas as pd
from collections import Counter
import re


st.set_page_config(page_title="Hotel Analytics", page_icon=":hotel:", layout="wide")

st.header("The Smart Tourism Development System is a data-driven solution enhancing Egypt’s tourism experience through analytics, machine learning")
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)
df = pd.read_csv("final_hotels_data.csv")

st.sidebar.image("logo.png" , width=200)
st.sidebar.write(":hotel: Hotel Data Analysis", )

st.sidebar.header("Select  Your Insights")
city = st.sidebar.multiselect("Select City", df["City"].unique())
if city:
    filtered_hotels = df[df["City"].isin(city)]["Hotel Name"].unique()
else:
    filtered_hotels = df["Hotel Name"].unique()
hotel = st.sidebar.multiselect("Select Hotel", filtered_hotels)
travel_type = st.sidebar.multiselect("Select Travel Type", df["Travel Type"].unique())

if city:
    df = df[df["City"].isin(city)]
if hotel:
    df = df[df["Hotel Name"].isin(hotel)]
if travel_type:
    df = df[df["Travel Type"].isin(travel_type)]



df_filtered = df[["Travel Type", "sentiment classification"]]
df_filtered["Sentiment"] = df_filtered["sentiment classification"].map({1: "Positive", -1: "Negative"})
df_grouped = df_filtered.groupby(["Travel Type", "Sentiment"]).size().reset_index(name="Review Count")





col1, col2 = st.columns(2)
with col1:
    st.subheader("Review Sentiment Analysis")
    sentiment_counts = df_filtered["Sentiment"].value_counts()
    fig2 = px.pie(
        names=sentiment_counts.index,
        values=sentiment_counts.values,
        title="Positive vs. Negative Review Distribution",
        color=sentiment_counts.index,
        color_discrete_map={"Positive": "red", "Negative": "green"})
    st.plotly_chart(fig2)




with col2:
    st.subheader("Sentiment by Travel Type")
    fig = px.bar(
        df_grouped,
        x="Travel Type",
        y="Review Count",
        color="Sentiment",
        barmode="stack",
        color_discrete_map={"Positive": "darkred", "Negative": "blue"},)
    fig.update_layout(
        title="Sentiment by Travel Type",
        xaxis_title="Travel Type",
        yaxis_title="Review Count",
        legend_title="Sentiment")
    st.plotly_chart(fig)


col3, col4 = st.columns(2)

with col3:
    negative_words_list = set([   
    "bad", "worst", "dirty", "rude", "expensive", "slow", "uncomfortable",
    "noisy", "cold", "old", "poor", "broken", "horrible", "awful", "smelly",
    "terrible", "unfriendly", "small", "boring", "disgusting", "overpriced"
])

    negative_reviews = df["Negative Review"].dropna()
    negative_reviews = negative_reviews[negative_reviews.str.lower() != "no negative feedback"]

    words = []
    for review in negative_reviews:
        word_list = re.findall(r'\b\w+\b', review.lower())  # تقسيم النص إلى كلمات
        filtered_words = [word for word in word_list if word in negative_words_list]  # الاحتفاظ بالكلمات السلبية فقط
        words.extend(filtered_words)

    word_counts = Counter(words)
    common_words = word_counts.most_common(10)  # استخراج أكثر 10 كلمات سلبية شيوعًا

    negative_words_df = pd.DataFrame(common_words, columns=["Keyword", "Frequency"])

    fign = px.bar(
        negative_words_df,
        x="Frequency",
        y="Keyword",
        orientation="h",
        title="Top 10 Negative Keywords from Reviews",
        color="Frequency",
        color_continuous_scale="reds",
    )
    st.plotly_chart(fign)


with col4:
    positive_words_list = set([
    "good", "great", "clean", "friendly", "comfortable", "excellent", "amazing",
    "helpful", "spacious", "quiet", "modern", "nice", "wonderful", "perfect",
    "fantastic", "cozy", "lovely", "awesome", "beautiful", "affordable"])

    positive_reviews = df["Positive Review"].dropna()
    positive_reviews = positive_reviews[positive_reviews.str.lower() != "no positive feedback"]

    words = []
    for review in positive_reviews:
        word_list = re.findall(r'\b\w+\b', review.lower())  # تقسيم النص إلى كلمات
        filtered_words = [word for word in word_list if word in positive_words_list]  # الاحتفاظ بالكلمات الإيجابية فقط
        words.extend(filtered_words)

    word_counts = Counter(words)
    common_words = word_counts.most_common(10)  # استخراج أكثر 10 كلمات إيجابية شيوعًا

    positive_words_df = pd.DataFrame(common_words, columns=["Keyword", "Frequency"])

    figp = px.bar(
        positive_words_df,
        x="Frequency",
        y="Keyword",
        orientation="h",
        title="Top 10 Positive Keywords from Reviews",
        color="Frequency",
        color_continuous_scale="blues",
    )
    st.plotly_chart(figp)