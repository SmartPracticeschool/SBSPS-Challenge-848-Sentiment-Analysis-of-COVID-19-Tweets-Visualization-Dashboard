from django.shortcuts import render
from django.http import HttpResponse
import tweepy
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics,svm
from sklearn.linear_model import LogisticRegression
import numpy as np
import sklearn
import pandas as pd
from sklearn.linear_model import LinearRegression
import re
import nltk
from textblob.en import sentiment
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))

# keys and tokens from the Twitter Dev Console - Authentication

consumerKey = '10gbePnFAmayg1E6zfEzhP7ZE'
consumerSecret = 'ChqP3PyDq7Fyqd1oivTsrHJIjQZPRxH70v60o5aU0bNQDrlp1M'
accessToken = '2290887223-4wXUmD5HSV4LPFbuYTiiNlP8kfgEGaTFPYs76gK'
accessTokenSecret = 'uYzfdnFkxNxNiaTM0AsDiDlL4rrdyxAEd8fUPVfB31p1I'

# establish connection with  twitter api
auth = tweepy.OAuthHandler(consumerKey,consumerSecret)
auth.set_access_token(accessToken, accessTokenSecret)

# connection with twitter
api = tweepy.API(auth)

#Handling Page Not found----
def handler404(request, exception):
    return render(request,'404.html')

#Handling Server Error-----
def handler500(request):
    return render(request, '500.html', status=500)

#Redirecting to about page--
def about(request):
    return render(request,'about.html')

'''Cleaning the tweets. Creating the function. Removing #, RT, @mentions'''
def clean_text(text):
    text = text.apply(lambda x : x.lower())
    text = text.apply(lambda x : x.strip())
    text = text.apply(lambda x: re.sub(" +"," ", x))
    text = text.apply(lambda x: re.sub(r"[-()\"#/@;:{}`+=~|!.*?,'0-9]", "",x))
    text = text.apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    return(text)

'''Create a function to compute positive,negative and neutral analysis'''
def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'

def getlabel(label):
    if label < 0:
        return -1
    elif label == 0:
        return 0
    else:
        return 1

'''function to count word frequency of thw words in the tweets'''
def hashtag_extract(x):
    hashtags = []
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"(\w+)", i)
        hashtags.append(ht)
    return hashtags

'''
 Training the model
 Classification is the process of predicting the class of given data points.
 train_model function can be used to train a model.
 Using the input passeds, the model is trained and accuracy score is computed.    
'''
def train_model(classifier, feature_vector_train, label, feature_vector_valid,  valid_y):
    classifier.fit(feature_vector_train, label)    # fit the training dataset on the classifier
    predictions = classifier.predict(feature_vector_valid)   # predict the labels on validation dataset
    return metrics.accuracy_score(classifier.predict(feature_vector_train), label), metrics.accuracy_score(predictions, valid_y)


# calculating percentage of respective sentiments(positive, negative, neutral)-------
def percentage(part, whole):
    return 100 * float(part) / float(whole)

def index(request):
    #Searching for tweets with covid19----
    tweets = tweepy.Cursor(api.search, q="#covid19").items(100)

    # add multiple arguments like range of date set limit etc
    #initializing variables to perform sentiment analysis---------

    positive = 0
    neutral = 0
    negative = 0
    polarity = 0  # for calculating average results of tweets call
    subjectivity = 0
    neutral_sentiment= []
    positive_sentiment = []
    negative_sentiment = []
    polarity_sentiment = []
    subjectivity_sentiment = []
    time_sentiment = []

    #Cleaning Data - Removing retweets--------
    search = "#covid19" + " -filter:retweets"
    tweets = tweepy.Cursor(api.search, q=search, lang = 'en').items(100)
    #t=[tweet.text for tweet in tweets]

    #Getting geographic location and timing of tweets-----
    tweets = tweepy.Cursor(api.search, q=search, lang = 'en').items(100)
    user_tweet_location = [[tweet.user.screen_name,tweet.text,tweet.user.location,tweet.created_at] for tweet in tweets]  #tweet.user.screen_name,
    tweets = tweepy.Cursor(api.search, q=search,place_country = 'IN', lang = 'en').items(100)

    #creating Dataframe---------
    df = pd.DataFrame(data=user_tweet_location,columns=['User',"Tweet","Location","Time"])
    df['Tweet'] = clean_text(df['Tweet'])


    #Traversing through each tweets time
    for i in df["Time"]:
        time= i
        time_sentiment.append(time)

    #traversing to each tweet---------

    for tweet in df.Tweet:
        print(tweet)
        analysis = TextBlob(tweet)
        polarity= analysis.sentiment.polarity   #calculating polarity
        subjectivity = analysis.sentiment.subjectivity   #calculating subjectivity
        polarity = round(polarity,3)
        subjectivity = round(subjectivity,3)
        subjectivity_sentiment.append(subjectivity)
        polarity_sentiment.append(polarity)

        #calculating polarity of tweets---------

        if (analysis.sentiment.polarity == 0):
            neutral += 1
            neutral_sentiment.append(analysis.sentiment.polarity)
        elif (analysis.sentiment.polarity > 0.00):
            positive += 1
            positive_sentiment.append(analysis.sentiment.polarity)

        elif (analysis.sentiment.polarity < 0.00):
            negative += 1
            negative_sentiment.append(analysis.sentiment.polarity)

    df['Polarity'] = polarity_sentiment
    df['Subjectivity'] = subjectivity_sentiment

    print("neutral_sentiment",neutral_sentiment)
    print("positive sentiment",positive_sentiment)
    print("negative sentiment",negative_sentiment)
    print("Polarity Sentiment", polarity_sentiment)

    #calculate percentage according to polarity
    #calling percentage method----------

    neutral1 = percentage(neutral,100)
    positive1 = percentage(positive,100)
    negative1 = percentage(negative,100)
    polarity = percentage(polarity,100)

    #for 2 decimal place

    neutral = format(neutral, '.2f')
    positive = format(positive, '.2f')
    negative = format(negative, '.2f')

    #creating the list of sentiment.
    senti = ['neutral','positive','negative']
    sizes = [neutral1,positive1,negative1]

    df['Analysis']=df['Polarity'].apply(getAnalysis)

    '''Calling a function getlabel() to get label 1 for positive,-1 for negative and 0 for neutral sentiment'''

    df['Label']=df['Polarity'].apply(lambda x : getlabel(x))

    #removing neutral tweets as they are not usefull for our prediction and they cause imbalancy of dataset
    #Storing the values in the new dataframe

    df2 = df[(df.Label!=0)]
    print(df2.head())

    #using numpy to count unique labels(tweets).
    unique,count = np.unique(df2.Label,return_counts=True)
    print(unique,count)

    #Sentiment prediction
    x = df2.Tweet
    y = df2.Label
    print(df2.head())

    #word frequency---------
    '''calling hashtag_extract() to count frequency of words'''
    # extracting frequency of words in positive tweets---
    positive_word_freq= hashtag_extract(df['Tweet'][df['Label'] == 1])

    # extracting frequency of words in negative tweets----
    negative_word_freq= hashtag_extract(df['Tweet'][df['Label'] == -1])

    # unnesting list - creating list of word frequencies---
    positive_word_freq = sum(positive_word_freq,[])
    negative_word_freq = sum(negative_word_freq,[])

    # word count of positive tweets----
    a = nltk.FreqDist(positive_word_freq)
    data1 = list(a.keys())
    count1 = list(a.values())

    # Paasing the word and its frequency in a dataframe------
    pos_df = pd.DataFrame({'Hashtag': data1,'Count': count1})
    # selecting top 10 most frequent hashtags
    pos_df = pos_df.nlargest(columns="Count", n = 10)
    hash = pos_df['Hashtag'].tolist()
    freq = pos_df['Count'].tolist()

    # word count of negative tweets----
    b = nltk.FreqDist(negative_word_freq)
    data2 = list(b.keys())
    count2 = list(b.values())

    neg_df = pd.DataFrame({'Hashtag1': data2,'Count2': count2})
    # selecting top 10 most frequent hashtags
    neg_df = neg_df.nlargest(columns="Count2", n = 10)
    hash1 = neg_df['Hashtag1'].tolist()
    freq1 = neg_df['Count2'].tolist()

    #Partitioning the data to perform prediction
    #Predictive analytics------

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.40,random_state=0)

    print("x Training data",x_train)
    #CountVectorizer for performing tokenization.
    #Scikit-learn’s CountVectorizer is used to convert a collection of text documents to a vector of token counts.

    # create a count vectorizer object
    vect = CountVectorizer(ngram_range=(1,1),binary=False,min_df=0.005,max_df=0.95)    #ngram_range are the combination of n terms together.
    vect.fit(x_train)  #fitting the training data.

    # transform the training and validation data using count vectorizer object
    x_train_DTM = vect.transform(x_train)
    x_test_DTM = vect.transform(x_test)

    #creating the pandas dataframe for the training data-------------------
    df_train = pd.DataFrame(x_train_DTM.toarray(),columns=vect.get_feature_names())

    '''Training the model-----
    #Using machine learning model to train final model.
    #We implement Naive Bayes Classifier and Support Vector Machine.'''

    '''finding the best algorithm to get the highest accuracy'''

    accuracy_L1 = train_model(MultinomialNB(), x_train_DTM, y_train,x_test_DTM, y_test)
    print("NB  for L1, Count Vectors: ", accuracy_L1)

    ''' SVM is a supervised machine learning algorithm which can be used for both classification or regression challenges.'''

    accuracy_L2 = train_model(svm.SVC(),   x_train_DTM, y_train,x_test_DTM, y_test)
    print("SVC  for L1, Count Vectors: ", accuracy_L2)

    #Logistic regression measures the relationship between the categorical dependent variable .&
    #one or more independent variables by estimating probabilities using a logistic function.

    lr = LogisticRegression()
    lr.fit(x_train_DTM, y_train)#fitting the training data.
    pred = lr.predict(x_test_DTM) #predicting the data.
    prediction = pred
    #Calculating the accuracy of training and testing data.
    prediction= list(prediction)
    print("Predicted data",prediction)
    y_test = y_test.to_numpy()
    print("Y test",y_test)
    train_acc = metrics.accuracy_score(lr.predict(x_train_DTM),  y_train)
    test_acc = metrics.accuracy_score(pred,y_test)
    print("Training Accuracy",train_acc)
    print("Testing Accuracy",test_acc)

    #Converting time_sentiment into type string---
    time_sentiment = list(map(str, time_sentiment))

    #sending the data to the dashboard.
    context = {'searchTerm':"#covid19",'noOfSearchTerms':100,'sizes':sizes,'senti':senti,'pred':prediction,'y_test':y_test,
               'polarity_sentiment':polarity_sentiment,'time_sentiment':time_sentiment,'hash':hash,'freq':freq,'hash1':hash1,'freq1':freq1}
    return render(request,'index.html',context)

def search(request):

    query1 = request.GET.get('query')
    print("query1",query1)

    # what and how many tweets to analyze---------
    #Searching for tweets----

    tweets = tweepy.Cursor(api.search, q=query1).items(100)

    # add multiple arguments like range of date set limit etc
    #initializing variables to perform sentiment analysis---------
    positive = 0
    neutral = 0
    negative = 0
    polarity = 0  # for calculating average results of tweets call
    subjectivity = 0
    neutral_sentiment= []
    positive_sentiment = []
    negative_sentiment = []
    polarity_sentiment = []
    subjectivity_sentiment = []
    time_sentiment = []

    #Cleaning Data - Removing retweets--------
    search = query1 + " -filter:retweets"
    search
    tweets = tweepy.Cursor(api.search, q=search, lang = 'en').items(100)

    #Getting geographic location of tweets-----
    tweets = tweepy.Cursor(api.search, q=search, lang = 'en').items(100)
    user_tweet_location = [[tweet.user.screen_name,tweet.text,tweet.user.location,tweet.created_at] for tweet in tweets]  #tweet.user.screen_name,
    #print("Loc",user_tweet_location)

    tweets = tweepy.Cursor(api.search, q=search,country = 'INDIA', lang = 'en').items(100)

    #creating Dataframe---------
    df = pd.DataFrame(data=user_tweet_location,columns=['User',"Tweet","Location","Time"])

    if df.empty:
        return HttpResponse("Search term does not exist")
    # Cleaning the tweets
    df['Tweet'] = clean_text(df['Tweet'])
    print(df['Tweet'])
    # Traversing through each tweets time
    for i in df["Time"]:
        time = i
        time_sentiment.append(time)

    # traversing to each tweet---------

    for tweet in df.Tweet:
        print(tweet)
        analysis = TextBlob(tweet)
        polarity = analysis.sentiment.polarity  # calculating polarity
        subjectivity = analysis.sentiment.subjectivity  # calculating subjectivity
        polarity = round(polarity, 3)
        subjectivity = round(subjectivity, 3)
        subjectivity_sentiment.append(subjectivity)
        polarity_sentiment.append(polarity)

        # calculating polarity of tweets---------

        if (analysis.sentiment.polarity == 0):
            neutral += 1
            neutral_sentiment.append(analysis.sentiment.polarity)
        elif (analysis.sentiment.polarity > 0.00):
            positive += 1
            positive_sentiment.append(analysis.sentiment.polarity)

        elif (analysis.sentiment.polarity < 0.00):
            negative += 1
            negative_sentiment.append(analysis.sentiment.polarity)

    df['Polarity'] = polarity_sentiment
    df['Subjectivity'] = subjectivity_sentiment

    print("neutral_sentiment", neutral_sentiment)
    print("positive sentiment", positive_sentiment)
    print("negative sentiment", negative_sentiment)
    print("Polarity Sentiment", polarity_sentiment)

    # calculate percentage according to polarity
    # calling percentage method----------

    neutral1 = percentage(neutral, 100)
    positive1 = percentage(positive, 100)
    negative1 = percentage(negative, 100)
    polarity = percentage(polarity, 100)

    # for 2 decimal place

    neutral = format(neutral, '.2f')
    positive = format(positive, '.2f')
    negative = format(negative, '.2f')

    # creating the list of sentiment.
    senti = ['neutral', 'positive', 'negative']
    sizes = [neutral1, positive1, negative1]

    '''Calling getAnalysis() to compute positive,negative and neutral analysis'''
    df['Analysis'] = df['Polarity'].apply(getAnalysis)


    '''Calling getlabel() to get label 1 for positive,-1 for negative and 0 for neutral sentiment'''
    df['Label'] = df['Polarity'].apply(lambda x: getlabel(x))

    ''' removing neutral tweets as they are not usefull for our prediction and they cause imbalancy of dataset . Storing the values in the new dataframe'''

    df2 = df[(df.Label != 0)]
    print(df2.head())

    # using numpy to count unique labels(tweets).
    unique, count = np.unique(df2.Label, return_counts=True)
    print(unique, count)

    # Sentiment prediction
    x = df2.Tweet
    y = df2.Label

    # word frequency---------
    # calling hashtag_extract to count word frequency of thw words in the tweets

    # extracting frequency of words in positive tweets---
    positive_word_freq = hashtag_extract(df['Tweet'][df['Label'] == 1])

    # extracting frequency of words in negative tweets----
    negative_word_freq = hashtag_extract(df['Tweet'][df['Label'] == -1])

    # unnesting list - creating list of word frequencies---
    positive_word_freq = sum(positive_word_freq, [])
    negative_word_freq = sum(negative_word_freq, [])
    print("Positive word freq",positive_word_freq)
    # word count of positive tweets----
    a = nltk.FreqDist(positive_word_freq)
    data1 = list(a.keys())
    count1 = list(a.values())

    # Paasing the word and its frequency in a dataframe------
    pos_df = pd.DataFrame({'Hashtag': data1, 'Count': count1})
    # selecting top 10 most frequent hashtags
    pos_df = pos_df.nlargest(columns="Count", n=10)
    hash = pos_df['Hashtag'].tolist()
    freq = pos_df['Count'].tolist()

    # word count of negative tweets----
    b = nltk.FreqDist(negative_word_freq)
    data2 = list(b.keys())
    count2 = list(b.values())

    neg_df = pd.DataFrame({'Hashtag1': data2, 'Count2': count2})
    # selecting top 10 most frequent hashtags
    neg_df = neg_df.nlargest(columns="Count2", n=10)
    hash1 = neg_df['Hashtag1'].tolist()
    freq1 = neg_df['Count2'].tolist()

    # Partitioning the data to perform prediction
    # Predictive analytics------
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.40, random_state=0)

    print("x Training data", x_train)
    print("Y test data", y_test)

    # CountVectorizer for performing tokenization.
    # Scikit-learn’s CountVectorizer is used to convert a collection of text documents to a vector of token counts.

    # create a count vectorizer object
    vect = CountVectorizer(ngram_range=(1, 1), binary=False, min_df=0.005,
                           max_df=0.95)  # ngram_range are the combination of n terms together.
    vect.fit(x_train)  # fitting the training data.

    # transform the training and validation data using count vectorizer object
    x_train_DTM = vect.transform(x_train)
    x_test_DTM = vect.transform(x_test)

    # creating the pandas dataframe for the training data-------------------
    df_train = pd.DataFrame(x_train_DTM.toarray(), columns=vect.get_feature_names())

    # Training the model-----

    # Using machine learning model to train final model.
    # We implement Naive Bayes Classifier and Support Vector Machine.

    # finding the best algorithm to get the highest accuracy

    accuracy_L1 = train_model(MultinomialNB(), x_train_DTM, y_train, x_test_DTM, y_test)
    print("NB  for L1, Count Vectors: ", accuracy_L1)

    # SVM is a supervised machine learning algorithm.
    # which can be used for both classification or regression challenges.

    accuracy_L2 = train_model(svm.SVC(), x_train_DTM, y_train, x_test_DTM, y_test)
    print("SVC  for L1, Count Vectors: ", accuracy_L2)

    # Logistic regression measures the relationship between the categorical dependent variable .&
    # one or more independent variables by estimating probabilities using a logistic function.

    lr = LogisticRegression()
    lr.fit(x_train_DTM, y_train)  # fitting the training data.
    pred = lr.predict(x_test_DTM)  # predicting the data.
    prediction = pred
    # Calculating the accuracy of training and testing data.
    prediction = list(prediction)
    print("Predicted data", prediction)
    y_test = y_test.to_numpy()
    train_acc = metrics.accuracy_score(lr.predict(x_train_DTM), y_train)
    test_acc = metrics.accuracy_score(pred, y_test)
    print("Training Accuracy", train_acc)
    print("Testing Accuracy", test_acc)
    #Prediction()
    # Converting time_sentiment into type string---
    time_sentiment = list(map(str, time_sentiment))

    #sending the data to the dashboard.
    #context = {'searchTerm':query1,'noOfSearchTerms':100,'sizes':sizes,'senti':senti,'pred':pred,'y_test':y_test}
    context = {'searchTerm':query1,'noOfSearchTerms':100,'sizes':sizes,'senti':senti,'pred':prediction,'y_test':y_test,
               'polarity_sentiment':polarity_sentiment,'time_sentiment':time_sentiment,'hash':hash,'freq':freq,'hash1': hash1,'freq1':freq1, 'search':query1}
    return render(request,'index.html',context)

