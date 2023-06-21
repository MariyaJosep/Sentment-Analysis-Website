

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
import helper_functions as hf

#To Hide Warnings
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)
# Viz Pkgs
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.use('Agg')
import seaborn as sns
#sns.set_style('darkgrid')


STYLE = """
<style>
img {
    max-width: 100%;
}
</style> """

def main():
    """ Common ML Dataset Explorer """
    #st.title("Live twitter Sentiment analysis")
    #st.subheader("Select a topic which you'd like to get the sentiment analysis on :")

    html_temp = """
	<div style="background-color:tomato;"><p style="color:white;font-size:40px;padding:9px">Live twitter Sentiment analysis</p></div>
	"""
    st.markdown(html_temp, unsafe_allow_html=True)
    st.subheader("Select a topic which you'd like to get the sentiment analysis on :")

    ################# Twitter API Connection #######################
    consumer_key = "gGNyUFA207t0IDsfgt6yK8ZAA"
    consumer_secret = "pgeBJQDx66h5sDSmA9GWjtLFVbnj1K7609B0UV3h5WPCpYQOZ7"
    access_token = "3225008987-pXhwCWb9rWcMDpaxCE7GSAr8hjNfsdTbIg4Cqcw"
    access_token_secret = "BTfxq05oASw2g5NW0WbcVoIXZcSapJEGc6wLTPwBfGEzr"



    # Use the above credentials to authenticate the API.

    auth = tweepy.OAuthHandler( consumer_key , consumer_secret )
    auth.set_access_token( access_token , access_token_secret )
    api = tweepy.API(auth)
    ################################################################
    
    df = pd.DataFrame(columns=["Date","User","IsVerified","Tweet","Likes","RT",'User_location'])
    
    # Write a Function to extract tweets:
    def get_tweets(Topic,Count,to_date):
        i=0
        #my_bar = st.progress(100) # To track progress of Extracted tweets
        print(to_date)
        #Returns tweets created before the given date. Date should be formatted as YYYY-MM-DD. Keep in mind that the search index has a 7-day limit. In other words, no tweets will be found for a date older than one week.
        #Returns tweets created before the given date. Date should be formatted as YYYY-MM-DD. Keep in mind that the search index has a 7-day limit. In other words, no tweets will be found for a date older than one week.
        for tweet in api.search_tweets(q=Topic,count=Count, lang="en",until=to_date):
            #time.sleep(0.1)
            #my_bar.progress(i)
            df.loc[i,"Date"] = tweet.created_at
            df.loc[i,"User"] = tweet.user.name
            df.loc[i,"IsVerified"] = tweet.user.verified
            df.loc[i,"Tweet"] = tweet.text
            df.loc[i,"Likes"] = tweet.favorite_count
            df.loc[i,"RT"] = tweet.retweet_count
            df.loc[i,"User_location"] = tweet.user.location
            #df.to_csv("TweetDataset.csv",index=False)
            #df.to_excel('{}.xlsx'.format("TweetDataset"),index=False)   ## Save as Excel
            i=i+1
            if i>Count:
                break
            else:
                pass
    # Function to Clean the Tweet.
    def clean_tweet(tweet):
        return ' '.join(re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|([RT])', ' ', tweet.lower()).split())
    
        
    # Funciton to analyze Sentiment
    def analyze_sentiment(tweet):
        analysis = TextBlob(tweet)
        if analysis.sentiment.polarity > 0:
            return 'Positive'
        elif analysis.sentiment.polarity == 0:
            return 'Neutral'
        else:
            return 'Negative'
    
    #Function to Pre-process data for Worlcloud
    def prepCloud(Topic_text,Topic):
        Topic = str(Topic).lower()
        Topic=' '.join(re.sub('([^0-9A-Za-z \t])', ' ', Topic).split())
        Topic = re.split("\s+",str(Topic))
        stopwords = set(STOPWORDS)
        stopwords.update(Topic) ### Add our topic in Stopwords, so it doesnt appear in wordClous
        ###
        text_new = " ".join([txt for txt in Topic_text.split() if txt not in stopwords])
        return text_new
    
    
    image = Image.open('TwitterSentimentAnalysis/Logo1.jpg')
    st.image(image, caption='Twitter for Analytics',use_column_width=True)
    
    
    # Collect Input from user :
   # Topic = str()
    #Count=str()
    #from_date=str()
   # to_date=str()
    Count=st.text_input("Enter the count")
    to_date=st.text_input("Enter the date (Returns tweets created on or before the given date. Date should be formatted as YYYY-MM-DD. Keep in mind that the search index has a 7-day limit. In other words, no tweets will be found for a date older than one week.)")
    Topic = str(st.text_input("Enter the topic you are interested in (Press Enter once done)"))     
    
    if len(Topic) > 0 :
        
        # Call the function to extract the data. pass the topic and filename you want the data to be stored in.
        with st.spinner("Please wait, Tweets are being extracted"):
            get_tweets(Topic , int(Count), to_date)
            df['Tweet']=df['Tweet'].apply(str.strip)
            print(df['Tweet'])
            df.drop_duplicates(subset=['Tweet'],inplace=True,keep='first')
            df.reset_index(inplace=True)

           #df['count']=df['Tweet'].apply(len)
            st.session_state['df']=df

            print(st.session_state.df)
        st.success('Tweets have been Extracted !!!!')    

           
    
        # Call function to get Clean tweets
        df['clean_tweet'] = df['Tweet'].apply(lambda x : clean_tweet(x))
    
        # Call function to get the Sentiments
        df["Sentiment"] = df["Tweet"].apply(lambda x : analyze_sentiment(x))
        
        
        # Write Summary of the Tweets
        st.write("Total Tweets Extracted for Topic '{}' are : {}".format(Topic,len(df.Tweet)))
        st.write("Total Positive Tweets are : {}".format(len(df[df["Sentiment"]=="Positive"])))
        st.write("Total Negative Tweets are : {}".format(len(df[df["Sentiment"]=="Negative"])))
        st.write("Total Neutral Tweets are : {}".format(len(df[df["Sentiment"]=="Neutral"])))
        
        # See the Extracted Data : 
        if st.button("See the Extracted Data"):
            #st.markdown(html_temp, unsafe_allow_html=True)
            st.success("Below is the Extracted Data :")
            st.write(df)
        
        
        # get the countPlot
        if st.button("Get Count Plot for Different Sentiments"):
            st.success("Generating A Count Plot")
            st.subheader(" Count Plot for Different Sentiments")
            st.write(sns.countplot(df["Sentiment"].value_counts()))
            st.pyplot()
        
        # Piechart 
        if st.button("Get Pie Chart for Different Sentiments"):
            st.success("Generating A Pie Chart")
            a=len(df[df["Sentiment"]=="Positive"])
            b=len(df[df["Sentiment"]=="Negative"])
            c=len(df[df["Sentiment"]=="Neutral"])
            d=np.array([a,b,c])
            explode = (0.1, 0.0, 0.1)
            st.write(plt.pie(d,shadow=True,explode=explode,labels=["Positive","Negative","Neutral"],autopct='%1.2f%%'))
            st.pyplot()
            
            
        # get the countPlot Based on Verified and unverified Users
        if st.button("Get Count Plot Based on Verified and unverified Users"):
            st.success("Generating A Count Plot (Verified and unverified Users)")
            st.subheader(" Count Plot for Different Sentiments for Verified and unverified Users")
            st.write(sns.countplot(x=df["Sentiment"],hue=df.IsVerified))
            st.pyplot()
        
        
        ## Points to add 1. Make Backgroud Clear for Wordcloud 2. Remove keywords from Wordcloud
        
        
        # Create a Worlcloud
        if st.button("Get WordCloud for all things said about {}".format(Topic)):
            st.success("Generating A WordCloud for all things said about {}".format(Topic))
            text = " ".join(review for review in df.clean_tweet)
            stopwords = set(STOPWORDS)
            text_newALL = prepCloud(text,Topic)
            wordcloud = WordCloud(stopwords=stopwords,max_words=800,max_font_size=70).generate(text_newALL)
            st.write(plt.imshow(wordcloud, interpolation='bilinear'))
            st.pyplot()
        
        
        #Wordcloud for Positive tweets only
        if st.button("Get WordCloud for all Positive Tweets about {}".format(Topic)):
            st.success("Generating A WordCloud for all Positive Tweets about {}".format(Topic))
            text_positive = " ".join(review for review in df[df["Sentiment"]=="Positive"].clean_tweet)
            stopwords = set(STOPWORDS)
            text_new_positive = prepCloud(text_positive,Topic)
            #text_positive=" ".join([word for word in text_positive.split() if word not in stopwords])
            wordcloud = WordCloud(stopwords=stopwords,max_words=800,max_font_size=70).generate(text_new_positive)
            st.write(plt.imshow(wordcloud, interpolation='bilinear'))
            st.pyplot()
        
        
        #Wordcloud for Negative tweets only       
        if st.button("Get WordCloud for all Negative Tweets about {}".format(Topic)):
            st.success("Generating A WordCloud for all Positive Tweets about {}".format(Topic))
            text_negative = " ".join(review for review in df[df["Sentiment"]=="Negative"].clean_tweet)
            stopwords = set(STOPWORDS)
            text_new_negative = prepCloud(text_negative,Topic)
            #text_negative=" ".join([word for word in text_negative.split() if word not in stopwords])
            wordcloud = WordCloud(stopwords=stopwords,max_words=800,max_font_size=70).generate(text_new_negative)
            st.write(plt.imshow(wordcloud, interpolation='bilinear'))
            st.pyplot()

            
         ############################           GENERATING DASHBOARD               #########################################################
        if st.button("Get Dashboard of the twitter Analysis"):
            st.success("Generating A Dashboard")
            print(st.session_state.df)
            st.subheader(" Twitter Analysis Dashboard")
            if 'df' in st.session_state:

                def make_dashboard(df, bar_color, wc_color):
                     # first row
                      col1, col2, col3 = st.columns([28, 34, 38])
                      with col1:
                          sentiment_plot = hf.plot_sentiment(df)
                          sentiment_plot.update_layout(height=350, title_x=0.5)
                          st.plotly_chart(sentiment_plot, theme=None, use_container_width=True)
                      with col2:
                          top_unigram = hf.get_top_n_gram(df, ngram_range=(1, 1), n=10)
                          unigram_plot = hf.plot_n_gram(
                top_unigram, title="Top 10 Occuring Words", color=bar_color
            )
                          unigram_plot.update_layout(height=350)
                          st.plotly_chart(unigram_plot, theme=None, use_container_width=True)
                      with col3:
                          top_bigram = hf.get_top_n_gram(df, ngram_range=(2, 2), n=10)
                          bigram_plot = hf.plot_n_gram(
                top_bigram, title="Top 10 Occuring Bigrams", color=bar_color
            )
                          bigram_plot.update_layout(height=350)
                          st.plotly_chart(bigram_plot, theme=None, use_container_width=True)

                      # second row
                      col1, col2 = st.columns([60, 40])
                      with col1:
                          def sentiment_color(sentiment):
                              if sentiment == "Positive":
                                  return "background-color: #1F77B4; color: white"
                              else:
                                  return "background-color: #FF7F0E"

                          st.dataframe(
                    df[["Sentiment", "Tweet"]].style.applymap(
                    sentiment_color, subset=["Sentiment"]
                ),
                height=350,
            )
                      with col2:
                          wordcloud = hf.plot_wordcloud(df, colormap=wc_color)
                          st.pyplot(wordcloud)

                adjust_tab_font = """
    <style>
    button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
        font-size: 20px;
    }
    </style>
    """

                st.write(adjust_tab_font, unsafe_allow_html=True)

                tab1, tab2, tab3, tab4= st.tabs(["All", "Positive 😊", "Negative ☹️","Neutral 😑"])
                with tab1:
                    df = st.session_state.df
                    make_dashboard(df, bar_color="#54A24B", wc_color="Greens")
                with tab2:
                    df = st.session_state.df.query("Sentiment == 'Positive'")
                    make_dashboard(df, bar_color="#1F77B4", wc_color="Blues")
                with tab3:
                    df = st.session_state.df.query("Sentiment == 'Negative'")
                    make_dashboard(df, bar_color="#FF7F0E", wc_color="Oranges")
                with tab4:
                    df = st.session_state.df.query("Sentiment == 'Neutral'")
                    make_dashboard(df, bar_color="#FFC0CB", wc_color="pink")    
    #####################################################################################################################

        
        
        
        
    st.sidebar.header("About App")
    st.sidebar.info("A Twitter Sentiment analysis Project which will scrap twitter for the topic selected by the user. The extracted tweets will then be used to determine the Sentiments of those tweets. \
                    The different Visualizations will help us get a feel of the overall mood of the people on Twitter regarding the topic we select.")
    st.sidebar.text("Built with Streamlit")
    
    #st.sidebar.header("For Any Queries/Suggestions Please reach out at :")
    #st.sidebar.info("darekarabhishek@gmail.com")
    #st.sidebar.subheader("Scatter-plot setup")
    #box1 = st.sidebar.selectbox(label= "X axis")
    #box2 = st.sidebar.selectbox(label="Y axis")
    #sns.jointplot(x=box1, y= box2, data=df, kind = "reg", color= "red")
    #st.pyplot()

   



    if st.button("Exit"):
        st.balloons()



if __name__ == '__main__':
    main()

