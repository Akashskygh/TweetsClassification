import tweepy
import time
import pandas as pd
import config

bearer_token = config.bearer_token

client = tweepy.Client(bearer_token, wait_on_rate_limit=True)

tweets = []
for response in tweepy.Paginator(client.search_all_tweets, 
                                query = '@GovCanHealth -is:retweet lang:en is:reply',
                                tweet_fields = ['created_at','text'],
                                max_results=500):
    time.sleep(1)
    tweets.append(response)

result = []
user_dict = {}

# Loop through each response object
for response in tweets:
    for tweet in response.data:
        # Put all of the information we want to keep in a single dictionary for each tweet
        result.append({'created_at': tweet.created_at, 'text': tweet.text})

df = pd.DataFrame(result)
df.to_csv('rawTweets.csv')