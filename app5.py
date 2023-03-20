import snscrape.modules.twitter as sntwitter
from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import streamlit as st
import re
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import seaborn as sns
import langid
from langdetect import detect
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import datetime

def app():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("Twitter Sentiment Analyzer")
    st.title("🙂😑☹️")
    st.subheader('Welcome to my app! 👋')

    activities = ["Tweet Analyzer", "Fetch Tweets","Fetch How Many Tweets You Want"]

    choice = st.sidebar.selectbox("Select Your Activity", activities)

    if choice == "Tweet Analyzer":

        # st.subheader("Analyze The Tweets According To Your Topic")
        st.subheader("This app performs the following tasks :-")
        st.write("1. Fetches the 10 most recent tweets from the given twitter handle")
        st.write("2. Generates a Word Cloud")
        st.write("3. Performs Sentiment Analysis a displays it in form of a Bar Graph,Pie Chart and Scatter Plot")

        raw_text = st.text_area("Enter the topic to extract tweets(without @)")

        st.markdown("<--------     Also Do checkout the another cool thing from the sidebar")

        Analyzer_choice = st.selectbox("Select the Activities", ["Show Recent Tweets", "Generate WordCloud", "Visualize the Sentiment Analysis"])

        if st.button("Analyze"):

            if Analyzer_choice == "Show Recent Tweets":
                st.success("Fetching last 10 Tweets")

                def Show_Recent_Tweets(raw_text):

                 # Set up search query
                    search_query = f"from:{raw_text}"

                # Fetch tweets
                    tweets = sntwitter.TwitterSearchScraper(search_query).get_items()

                # Extract tweets
                    recent_tweets = []
                    for i, tweet in enumerate(tweets):
                        if i >= 10:
                            break
                        lang = detect(tweet.content)
                        if lang == 'en':

                            recent_tweets.append(tweet.content)

                    return recent_tweets

                recent_tweets = Show_Recent_Tweets(raw_text)

                st.write(recent_tweets)

            elif Analyzer_choice == "Generate WordCloud":
                 st.success("Generating Word Cloud")

                 def gen_wordcloud():

                    # Set up search query
                    search_query = f"from:{raw_text}"

                    # Fetch tweets
                    tweets = sntwitter.TwitterSearchScraper(search_query).get_items()

                    # Extract tweets
                    tweets_list = []
                    for i, tweet in enumerate(tweets):
                        if i >= 100:
                            break
                        lang = detect(tweet.content)
                        if lang == 'en':
                            tweets_list.append(tweet.content)

                    # Create a dataframe with a column called Tweets
                    df = pd.DataFrame(tweets_list, columns=['Tweets'])
                    #
                    # Clean the tweet texts
                    def cleanTxt(text):
                        text = re.sub('@[A-Za-z0–9]+', '', text) # Remove @mentions
                        text = re.sub('#', '', text) # Remove '#' hash tag
                        text = re.sub('RT[\s]+', '', text) # Remove RT
                        text = re.sub('https?:\/\/\S+', '', text) # Remove hyperlink
                        text = re.compile("["
                                           u"\U0001F600-\U0001F64F"  # emoticons
                                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                           u"\U00002702-\U000027B0"
                                           u"\U000024C2-\U0001F251"
                                           "]+", flags=re.UNICODE).sub(r'',text)
                        return text
                     # Apply the cleanTxt function to the 'Tweets' column
                    df['Tweets'] = df['Tweets'].str.replace('http\S+|www.\S+', '', case=False)
                    df['Tweets'] = df['Tweets'].str.replace('\n', ' ', case=False)
                    df['Tweets'] = df['Tweets'].str.replace('[^\w\s#@/:%.,_-]', '', flags=re.UNICODE)
                    df['Tweets'] = df['Tweets'].apply(cleanTxt)

                    # word cloud visualization
                    allWords = ' '.join([twts for twts in df['Tweets']])
                    wordCloud = WordCloud(width=500, height=300, random_state=21, max_font_size=110).generate(allWords)
                    plt.imshow(wordCloud, interpolation="bilinear")
                    plt.axis('off')
                    plt.savefig('WC.jpg')
                    img = Image.open("WC.jpg")
                    return img
                 img = gen_wordcloud()
                 st.image(img)

            else:
                def get_data(raw_text):
                    search_query = f"from:{raw_text}"
    
                            # Fetch tweets
                    tweets = sntwitter.TwitterSearchScraper(search_query).get_items()
    
                            # Extract tweets
                    rawContent = []
                    for i, tweet in enumerate(tweets):
                        if i >= 100:
                            break
                        lang = detect(tweet.content)
                        if lang == 'en':
                            rawContent.append(tweet.content)

                     # Create a dataframe with a column called Tweets
                    df = pd.DataFrame(rawContent, columns=['Tweets'])
    
                    def cleanTxt(text):
                          text = re.sub('@[A-Za-z0–9]+', '', text) #Removing @mentions
                          text = re.sub('#', '', text) # Removing '#' hash tag
                          text = re.sub('RT[\s]+', '', text) # Removing RT
                          text = re.sub('https?:\/\/\S+', '', text) # Removing hyperlink
                          text = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE).sub(r'',text)
                          return text

                    # Clean the tweets
                    df['Tweets'] = df['Tweets'].str.replace('http\S+|www.\S+', '', case=False)
                    df['Tweets'] = df['Tweets'].str.replace('\n', ' ', case=False)
                    df['Tweets'] = df['Tweets'].str.replace('[^\w\s#@/:%.,_-]', '', flags=re.UNICODE)
                    df['Tweets'] = df['Tweets'].apply(cleanTxt)

                    def getSubjectivity(text):
                            return TextBlob(text).sentiment.subjectivity

                         # Create a function to get the polarity
                    def getPolarity(text):
                        return  TextBlob(text).sentiment.polarity

                        # Create two new columns 'Subjectivity' & 'Polarity'
                    df['Subjectivity'] = df['Tweets'].apply(getSubjectivity)
                    df['Polarity'] = df['Tweets'].apply(getPolarity)

                    def getAnalysis(score):
                        if score < 0:
                            return 'Negative'
                        elif score == 0:
                            return 'Neutral'
                        else:
                            return 'Positive'

                    df['Analysis'] = df['Polarity'].apply(getAnalysis)
                    return df

                def plot_Analysis(df):
                   sns.countplot(x=df["Analysis"],data=df)
                   plt.title("Sentiment Analysis")
                   plt.xlabel("Sentiment")
                   plt.ylabel("Counts")
                   return plt
                
                if st.success("Generating Visualization for Sentiment Analysis"):
                    df = get_data(raw_text)
                    plot = plot_Analysis(df)
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.pyplot(plot,use_container_width=True)
                
                def plot_Analysis_1(df):
                    plt.figure(figsize=(10,6))
                    df['Analysis'].value_counts().plot(kind='pie',autopct='%1.0f%%',colors=["hotpink", "cyan", "lawngreen"])
                    plt.title('Sentiment Analysis')
                    plt.legend(['Positive', 'Neutral', 'Negative'])
                    plt.xlabel('')
                    plt.ylabel('')


                if st.success("Generating Visualization for Sentiment Analysis"):
                    df = get_data(raw_text)
                    plot = plot_Analysis_1(df)
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.pyplot(plot,use_container_width=True)
                
                def plot_polarity_vs_subjectivity(df):
                    plt.figure(figsize=(8,6))
    
                         # Assign colors based on sentiment label
                    Colors = []
                    for sentiment in df['Analysis']:
                        if sentiment == 'Positive':
                            Colors.append('green')
                        elif sentiment == 'Negative':
                            Colors.append('red')
                        else:
                            Colors.append('blue')
    
                    plt.scatter(df['Polarity'], df['Subjectivity'], c=Colors)
                    plt.title('Sentiment Analysis')
                    plt.xlabel('Polarity')
                    plt.ylabel('Subjectivity')
                    return plt

                
                if st.success("Generating Visualization for Sentiment Analysis"):
                      st.success("Generating Scatter Plot [positive]=green,[Negative]=red,[neutral]=blue")
                      df = get_data(raw_text)
                      plot = plot_polarity_vs_subjectivity(df)
                      st.set_option('deprecation.showPyplotGlobalUse', False)
                      st.pyplot(plot,use_container_width=True)
                   
                
    elif choice == "Fetch Tweets":

        st.subheader("This Tool Performs The Following Tasks:-")
        st.write("1. It fetches the last 100 tweets from the twitter")
        st.write("2. Converts It Into a DataFrame")
        st.write("3. Cleans The Text")
        st.write("4. Analyzes Subjectivity of tweets and adds an additional column for it")
        st.write("5. Analyzes Polarity of tweets and adds an additional column for it")
        st.write("6. Analyzes Sentiments and add column of tweets and Generate visualization")
        user_name = st.text_area("*Enter the topic to extract tweets(without @)*")
        st.markdown("<--------     Also Do checkout the another cool tool from the sidebar")

        def get_data(user_name):
            search_query = f"from:{user_name}"

                    # Fetch tweets
            tweets = sntwitter.TwitterSearchScraper(search_query).get_items()

                    # Extract tweets
            rawContent = []
            for i, tweet in enumerate(tweets):
                if i >= 100:
                    break
                lang = detect(tweet.content)
                if lang == 'en':
                    rawContent.append(tweet.content)

                        # Create a dataframe with a column called Tweets
            df = pd.DataFrame(rawContent, columns=['Tweets'])
            def cleanTxt(text):
                text = re.sub('@[A-Za-z0–9]+', '', text) #Removing @mentions
                text = re.sub('#', '', text) # Removing '#' hash tag
                text = re.sub('RT[\s]+', '', text) # Removing RT
                text = re.sub('https?:\/\/\S+', '', text) # Removing hyperlink
                text = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE).sub(r'',text)
                return text

                # Clean the tweets
            df['Tweets'] = df['Tweets'].str.replace('http\S+|www.\S+', '', case=False)
            df['Tweets'] = df['Tweets'].str.replace('\n', ' ', case=False)
            df['Tweets'] = df['Tweets'].str.replace('[^\w\s#@/:%.,_-]', '', flags=re.UNICODE)
            df['Tweets'] = df['Tweets'].apply(cleanTxt)


            def getSubjectivity(text):
                return TextBlob(text).sentiment.subjectivity

                    # Create a function to get the polarity
            def getPolarity(text):
                return  TextBlob(text).sentiment.polarity


                    # Create two new columns 'Subjectivity' & 'Polarity'
            df['Subjectivity'] = df['Tweets'].apply(getSubjectivity)
            df['Polarity'] = df['Tweets'].apply(getPolarity)

            def getAnalysis(score):
                if score < 0:
                    return 'Negative'

                elif score == 0:
                    return 'Neutral'


                else:
                    return 'Positive'

    
                        
            df['Analysis'] = df['Polarity'].apply(getAnalysis)
            return df
        
        def plot_Analysis(df):
           sns.countplot(x=df["Analysis"],data=df)
           plt.title("Sentiment Analysis")
           plt.xlabel("Sentiment")
           plt.ylabel("Counts")
           return plt

        if st.button("Show Data"):
            st.success("Fetching Last 100 Tweets")
            df=get_data(user_name)
            st.write(df)
            plot = plot_Analysis(df)
            st.pyplot(plot, use_container_width=True)

    else :
        st.subheader("Fetch Number Of Tweets ")
        st.write("1. By Default Starting Date Is Since:2022-01-01 ")
        st.write("2. Fetch Maximum 1000 Tweets ")
        st.write("3. Analyze Tweets and Adds An Sentiment_Score Column For It")
        st.write("4. Analyzes Of Tweets and Adds Sentiment Column For It Where You See Positive,Negitive & Neutral Sentiment")
        st.write("5. Generate Visualization( Pie Chart )")
        # Set up NLTK sentiment analyzer
        nltk.download('vader_lexicon')
        analyzer = SentimentIntensityAnalyzer()

                # Function to scrape tweets and perform sentiment analysis
                # def get_sentiments(query, num_tweets, start_date="2022-01-01", end_date=None):
        def get_sentiments(query, num_tweets):
            tweets_list = []
            
            # for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f"{query}  since:{start_date} until:{end_date}").get_items()):
            for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f"{query}  since:2022-01-01" ).get_items()):
                if i >= num_tweets:
                    break
                try:
                    # Check if the tweet is in English
                    if langid.classify(tweet.content)[0] != 'en':
                        continue
                except langid.langid.LanguageIdentifierException:
                    # If langid fails, assume the tweet is not in English
                        continue
                tweet_dict = {}
                tweet_dict['text'] = tweet.content
                tweet_dict['date'] = tweet.date.strftime('%Y-%m-%d %H:%M:%S')
                tweet_dict['sentiment_score'] = analyzer.polarity_scores(tweet_dict['text'])['compound']
        
        
                if tweet_dict['sentiment_score'] >= 0.05:

                    tweet_dict['sentiment'] = 'Positive'

                elif tweet_dict['sentiment_score'] <= -0.05:

                    tweet_dict['sentiment'] = 'Negative'

                else:
                    tweet_dict['sentiment'] = 'Neutral'

                tweets_list.append(tweet_dict)

            return tweets_list

            
        # st.title("Our App Will Do Sentiment Analysis Using NLTK")
        query = st.text_input("Enter query to search on Twitter without@")
        # start_date = st.text_input("Enter start date (format: yyyy-mm-dd))",value=None)
        # end_date = st.text_input("Enter end date (format: yyyy-mm-dd) (leave blank for current date)",value=None)
        num_tweets = st.number_input("Enter number of tweets to analyze", min_value=1, max_value=1000, value=100)
        
        # if start_date is not None and start_date != "":
        #     try:
        #         start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        #     except ValueError:
        #         # st.error("Please enter a valid start date in the format yyyy-mm-dd.")
        #         start_date = None

        # if end_date is None:
        #         # Use current date as end date if not provided
        #         end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        # else:
        #     try:
        #         end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        #     except ValueError:
        #                 # st.error("Please enter a valid end date in the format yyyy-mm-dd.")
        #         end_date = None

        # if start_date is not None and end_date is not None and start_date > end_date:
        #         st.error("Please enter a valid date range: end date should be greater than or equal to start date.")
        # else:
        #         num_tweets = st.number_input("Enter number of tweets to analyze", min_value=1, max_value=1000, value=100)

        # if start_date == "":
        #      # Use default start date if not provided
        #     start_date = "2022-01-01"
        if st.button("Show Data"):
            # if start_date is None or end_date is None:
            #     tweets_list = get_sentiments(query, num_tweets)
            # else:
            #     start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            #     end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
            tweets_list = get_sentiments(query, num_tweets)
            df = pd.DataFrame(tweets_list)
            st.write(df)


        
              
                    # Create pie chart of sentiment distribution
            fig = px.pie(df, values=df['sentiment'].value_counts(), names=df['sentiment'].value_counts().index)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(title="Sentiment Distribution of Tweets")
            st.plotly_chart(fig)
        
















if __name__ == "__main__":
    app()


st.subheader(' -------------------App By :  Shubham,Devesh,Dineshwari,Jaggupati,Himanshu------------------- :sunglasses:')


