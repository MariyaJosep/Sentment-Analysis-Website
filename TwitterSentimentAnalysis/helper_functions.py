import streamlit as st
import warnings
warnings.filterwarnings("ignore")
# EDA Pkgs
import pandas as pd
import numpy as np
import pandas as pd
import tweepy
import json
from tweepy import OAuthHandler
import re
import textblob
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import openpyxl
import time
import tqdm
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px
import plotly.io as pio
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.use('Agg')
import seaborn as sns
import App_Streamlit

def plot_sentiment(df):
        sentiment_count = df["Sentiment"].value_counts()
        fig = px.pie(
        values=sentiment_count.values,
        names=sentiment_count.index,
        hole=0.3,
        title="<b>Sentiment Distribution</b>",
        color=sentiment_count.index,
        color_discrete_map={"Positive": "#1F77B4", "Negative": "#FF7F0E", "Neutral":"#FFC0CB"},
    )
        fig.update_traces(
        textposition="inside",
        texttemplate="%{label}<br>%{value} (%{percent})",
        hovertemplate="<b>%{label}</b><br>Percentage=%{percent}<br>Count=%{value}",
    )
        fig.update_layout(showlegend=False)
        return fig


def plot_wordcloud(df, colormap="Greens"):
    stopwords = set()
    with open("static/en_stopwords_viz.txt", "r") as file:
        for word in file:
            stopwords.add(word.rstrip("\n"))

    stopwords=list(stopwords)   
    cmap = matplotlib.cm.get_cmap(colormap)(np.linspace(0, 1, 20))
    cmap = matplotlib.colors.ListedColormap(cmap[10:15])
    mask = np.array(Image.open("static/twitter_mask.png"))
    font = "static/quartzo.ttf"
    text = " ".join(df["clean_tweet"])
    wc = WordCloud(
        background_color="white",
        font_path=font,
        stopwords=stopwords,
        max_words=90,
        colormap=cmap,
        mask=mask,
        random_state=42,
        collocations=False,
        min_word_length=2,
        max_font_size=200,
    )
    wc.generate(text)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Wordcloud", fontdict={"fontsize": 16}, fontweight="heavy", pad=20, y=1.0)
    return fig


def get_top_n_gram(df, ngram_range, n=10):
    stopwords = set()
    with open("static/en_stopwords_viz.txt", "r") as file:
        for word in file:
            stopwords.add(word.rstrip("\n"))
    stopwords=list(stopwords)         
    corpus = df["clean_tweet"]
    vectorizer = CountVectorizer(
        analyzer="word", ngram_range=ngram_range, stop_words=stopwords
    )
    X = vectorizer.fit_transform(corpus.astype(str).values)
    words = vectorizer.get_feature_names_out()
    words_count = np.ravel(X.sum(axis=0))
    df = pd.DataFrame(zip(words, words_count))
    df.columns = ["words", "counts"]
    df = df.sort_values(by="counts", ascending=False).head(n)
    df["words"] = df["words"].str.title()
    return df


def plot_n_gram(n_gram_df, title, color="#54A24B"):
    fig = px.bar(
        x=n_gram_df.counts,
        y=n_gram_df.words,
        title="<b>{}</b>".format(title),
        text_auto=True,
    )
    fig.update_layout(plot_bgcolor="white")
    fig.update_xaxes(title=None)
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_traces(hovertemplate="<b>%{y}</b><br>Count=%{x}", marker_color=color)
    return fig
