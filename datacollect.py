import pandas as pd
import json
import io
import numpy as np
from pandas.io.json import json_normalize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



data_json = io.open('malalatweets.json', mode='r', encoding='utf-8').read()
data_python = json.loads(data_json)

df = json_normalize(data_python)
df.to_csv("outputtweets2.csv")

#read organised data into python to analyse
tw_data = pd.read_csv('outputtweets2.csv',lineterminator='\n')
tw_text = tw_data['text'].astype(str)
tw_time = tw_data['timestamp'].astype(str)
#print(tw_data['text'])

analyzer = SentimentIntensityAnalyzer()
neg_sentiment = []
pos_sentiment = []
neu_sentiment = []
comp_sentiment = []
for tweet in tw_text:
    vs = analyzer.polarity_scores(tweet)
    #print("{:-<65} {}".format(tweet, str(vs)))
    #print(vs['neg'])
    neg_sentiment = np.append(neg_sentiment, vs['neg'])
    pos_sentiment = np.append(pos_sentiment, vs['pos'])
    neu_sentiment = np.append(neu_sentiment, vs['neu'])
    comp_sentiment = np.append(comp_sentiment, vs['compound'])
#print(sentiment)

tw_time = tw_time.str.split("T", expand=True)
#print(tw_time)
tw_data['date'] = tw_time[0]
tw_data['time'] = tw_time[1]
tw_data['pos'] = pos_sentiment
tw_data['neg'] = neg_sentiment
tw_data['neu'] = neu_sentiment
tw_data['comp'] = comp_sentiment

tw_data.to_csv('sent_output2.csv')
