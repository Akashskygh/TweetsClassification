#!pip install snscrape
import snscrape.modules.twitter as sntwitter
import pandas as pd

query = "(to:GovCanHealth)"
tweets = []
limit = 20000

for tweet in sntwitter.TwitterSearchScraper(query).get_items():

    if len(tweets) == limit:
        break
    else:
        tweets.append([tweet.date,tweet.content,tweet.user.username,tweet.user.location,tweet.user.verified,tweet.user.followersCount, 
                        tweet.id,tweet.lang,tweet.hashtags,tweet.replyCount,tweet.retweetCount,tweet.likeCount,tweet.quoteCount,tweet.media])
        
df = pd.DataFrame(tweets, columns=['created_at', 'text', 'username', 'location', 'verified', 'followersCount', 'id', 'lang', 'hashtags', 
                                    'replyCount', 'retweetCount', 'likeCount', 'quoteCount', 'media'])
df.to_csv('rawTweets.csv')