# SBSPS-Challenge-848-Sentiment-Analysis-of-COVID-19-Tweets-Visualization-Dashboard
The Twitter Sentiment Analysis is a predictive model trained using Machine Learning. The project is developed in Python programming language with Python Django as a framework.
The data to be analysed and visualized are gathered from the twitter itself using the tweepy API.

The python code of the project is mentioned in the views.py file and the UI part is in index.html file.
1.Objective of the project- 
In view with the current scenario of the outbreak of the Covid-19 it is extremely necessary to have an application which can monitor and analyze the people's sentiments. In this twitter sentiment analysis, our main objective is to extract the sentiments of the people on the outbreak of Covid-19 and the extension of the lockdown. We will analyze this information and also visualize it graphically. Such information may be of high value to different sectors. Talking about the Government, the government will get to know the sentiments of its people and take necessary actions to handle the scenario. This may also help the government to enact better policies for the people of the country.
The benefit of such kind of twitter sentiment analysis is not restricted to only one sector but it might affect a number of people in a positive sense. Now considering the scenario of the insurance sector, the companies may launch new policies seeing the current market conditions. The health sector will get to know about the medical needs of the people of the country. Similarly, there are other sectors also which will be benefited with this twitter sentiment analysis. 
To achieve this goal we will be calling the twitter API (tweepy) and get the tweets of the people. As we are working on the sentimental analysis we will get the comments (sentiments) from the tweets. For authentication, we will import the OAuthHandler from the tweepy module and we will use TextBlob to tokenize the comments and NLTK for the analysis and extract the useful information from it.TextBlob aims to provide access to common text-processing operations through a familiar interface on which we will perform sentiment analysis. After getting the tweets tokenized, the polarity (positive, negative, neutral) will be calculated. Once all the useful data is extracted. For handling the huge amount of data pandas library will be used.
After all the analysis of data is done we use Python Django framework for visualizing the data in various ways. We represent the real-time data in the form of various kinds of graphs, charts, and statistics.


2. LITERATURE SURVEY

2.1 Existing Problem
      There are many websites which show twitter sentiment analysis but as one can find they do not represent the information in a clear manner as it is quite difficult to distinguish between the kind of sentiment. The biggest drawback of all these web applications is that they can analyse only the pre-searched keywords or #tags making the application useful to only the limited section of the society.

2.2 Proposed Solution
       In our twitter sentiment visualization, we took care of all these drawbacks and have represented the information in a very clear manner that even the person with non technical background will be able to easily monitor the sentiments. The biggest feature which we have added is that we have a search bar to allow our users to search for any relevant #tags at real-time.
       
3. PROJECT MODULES

1.Search #tag – The application allows its users to search for some specific #tag. Although analytics and visualization results on some of the #tags related to Covid-19 are already displayed on the main page. The data related to searched #tag will appear above the pre-displayed results.

2.Monitoring and filtering data – The data will be monitored on the real time basis. The data can also be filtered on the basis of the searched data.

3.Visualizing through graphs – The analysis will be performed on the basis of the tweets of the people fetched using the twitter.       
