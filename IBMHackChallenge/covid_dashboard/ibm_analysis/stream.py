# import matplotlib
# from Cython.Shadow import inline
# from tweepy import Stream
# from tweepy import OAuthHandler
# from tweepy.streaming import StreamListener
# import json
# import time
# import tweepy
# import pandas as pd
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
#
# # keys and tokens from the Twitter Dev Console - Authentication
# consumerKey = '10gbePnFAmayg1E6zfEzhP7ZE'
# consumerSecret = 'ChqP3PyDq7Fyqd1oivTsrHJIjQZPRxH70v60o5aU0bNQDrlp1M'
# accessToken = '2290887223-4wXUmD5HSV4LPFbuYTiiNlP8kfgEGaTFPYs76gK'
# accessTokenSecret = 'uYzfdnFkxNxNiaTM0AsDiDlL4rrdyxAEd8fUPVfB31p1I'
#
# # establish connection with  twitter api
#
# auth = tweepy.OAuthHandler(consumerKey,consumerSecret)
# auth.set_access_token(accessToken, accessTokenSecret)
#
# # connection with twitter
# api = tweepy.API(auth)
#
#
# def index(request):
#
#     class MyStreamListener(tweepy.StreamListener):
#
#         def __init__(self, api=None):
#             # inherit class attributes
#             super(MyStreamListener, self).__init__()
#             self.num_tweets = 0
#             self.file = open("tweets.txt", "w+")
#
#         def on_status(self, status):
#             tweet = status._json
#
#             self.file.write( json.dumps(tweet) + '\n' )
#
#             self.num_tweets += 1
#             if self.num_tweets < 1000:
#                 return True
#             else:
#                 return False
#             self.file.close()
#
#         def on_error(self, status):
#             print(status)
#     l = MyStreamListener()
#
#     # Create you Stream object with authentication
#     stream = tweepy.Stream(auth, l)
#
#
#     # Filter Twitter Streams to capture data by the keywords:
#     stream.filter(track=['Trump stupid','Trump Hillary','Hillary stupid','Trump daughter'])
#
#     # Initialize empty list to store tweets: tweets_data
#     tweets_data = []
#
#     # Open connection to file
#     h=open('tweets.txt','r')
#
#     # Read in tweets and store in list: tweets_data
#     for i in h:
#         try:
#             tmp=json.loads(i)
#             tweets_data.append(tmp)
#         except:
#             print ('X')
#     h.close()
#
#
#
#     pd.DataFrame(tweets_data).head(1)
#
#     # Build DataFrame of tweet texts and languages
#     df = pd.DataFrame(tweets_data, columns=['text', 'lang'])
#     print (df.shape)
#     # Print head of DataFrame
#     print(df.head(3))
#
#     import re
#     def word_in_text(word, tweet):
#         word = word.lower()
#         text = tweet.lower()
#         match = re.search(word, tweet)
#
#         if match:
#             return True
#         return False
#     # Initialize list to store tweet counts
#     [Trump, stupid, girl, hillary] = [0, 0, 0, 0]
#
#     # Iterate through df, counting the number of tweets in which
#     # each candidate is mentioned
#     for index, row in df.iterrows():
#         Trump += word_in_text('trump', row['text'].lower())
#         stupid += word_in_text('stupid', row['text'].lower())
#         girl += word_in_text('girl', row['text'].lower())
#         hillary += word_in_text('hillary', row['text'].lower())
#     print (Trump, stupid, girl, hillary)
#
#     df['text'].str.contains('hillary',case=False).sum()
#
#     #override tweepy.StreamListener to add logic to on_status
#     class test(tweepy.StreamListener):
#         def __init__(self):
#             # inherit class attributes
#             super(test, self).__init__()
#             #         tweepy.StreamListener.__init__(self)
#
#             self.num=0
#
#
#         def on_status(self, status):
#             self.num+=1
#             #print self.num
#             print(status.text)
#             if self.num==10:
#
#                 #returning False in on_data disconnects the stream
#                 return False
#
#         def on_error(self, status):
#             print(status)
#
#
#     # Initialize Stream listener
#     l = test()
#
#     # Create you Stream object with authentication
#     stream = tweepy.Stream(auth, l)
#
#     # Filter Twitter Streams to capture data by the keywords:
#     stream.filter(track=['Trump stupid','Trump Hillary','Hillary','Trump daughter'])
#
#     # Import packages
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#
#     #%matplotlib inline
#     # Set seaborn style
#     sns.set(color_codes=True)
#
#     # Create a list of labels:cd
#     cd = ['hillary', 'trump', 'stupid', 'girl']
#
#     # Plot histogram
#     ax = sns.barplot(cd, [hillary, Trump, stupid, girl],alpha=.6)
#     ax.set(ylabel="count")
#     plt.show()
#
#
#     return(request)
#
#
#
#
# '''
# def index(request):
#     class MyStreamListener(tweepy.StreamListener):
#
#         def on_status(self, status):
#             print(status.text)
#
#
#     myStreamListener = MyStreamListener()
#     myStream = tweepy.Stream(auth = api.auth, listener=myStreamListener)
#
#
#     myStream.filter(track=['covid19'], is_async=True)
#
#
#
#     class MyStreamListener(tweepy.StreamListener):
#
#         def on_error(self, status_code):
#             if status_code == 420:
#                 #returning False in on_error disconnects the stream
#                 return False
#
#             # returning non-False reconnects the stream, with backoff.
#     # add multiple arguments like range of date set limit etc
#     #initializing variables to perform sentiment analysis---------
#     positive = 0
#     neutral = 0
#     negative = 0
#     polarity = 0  # for calculating average results of tweets call
#     subjectivity = 0
#     neutral_sentiment= []
#     positive_sentiment = []
#     negative_sentiment = []
#     polarity_sentiment = []
#     subjectivity_sentiment = []
#
#
#
#     #Cleaning Data - Removing retweets--------
#     #search = searchTerm + " -filter:retweets"
#     #search
#     #tweets = tweepy.Cursor(api.search, q=search, lang = 'en').items(noOfSearchTerms)
#     #t=[tweet.text for tweet in tweets]
#
#     #Getting geographic location of tweets-----
#     #tweets = tweepy.Cursor(api.search, q=search, lang = 'en').items(noOfSearchTerms)
#    ''' user_tweet_location = [[tweet.user.screen_name,tweet.text,tweet.user.location] for tweet in myStream]  #tweet.user.screen_name,
#     print("Loc",user_tweet_location)
#
#     #tweets = tweepy.Cursor(api.search, q=search,place_country = 'IN', lang = 'en').items(noOfSearchTerms)
#
#     #creating Dataframe---------
#     df = pd.DataFrame(data=user_tweet_location,columns=['User',"Tweet","Location"])
#     print(df.head()
#     print(df['Tweet'])
#
# '''
#     class listener(StreamListener):
#
#         def on_data(self, data, analyzer=None):
#             try:
#                 data = json.loads(data)
#                 tweet = (data['text'])
#                 time_ms = data['timestamp_ms']
#                 vs = analyzer.polarity_scores(tweet)
#                 sentiment = vs['compound']
#                 print(time_ms, tweet, sentiment)
#
#             except KeyError as e:
#                 print(str(e))
#             return(True)
#
#             return(request)'''