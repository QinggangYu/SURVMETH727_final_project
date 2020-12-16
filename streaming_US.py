#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 16:51:34 2020

@author: qinggang
"""

import tweepy
import pandas as pd
import time
import os
import json
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
import numpy as np

os.chdir('/Users/Qinggang1/Documents/graduate/Survmeth727/Final_project')

auth = tweepy.OAuthHandler('xRAQQuBmhwlJHEHITkWSsKgAy', 
                           'iygqhDuuLVXWVbq5hXg3VTHFZoj8Yvnr6XIFY4WMsOp1CJ93am')
auth.set_access_token('1471316202-GZfO1eNAcS1KC3fSU0mxcc11llyKOAbBhfvHuua', 
                      'vicUgD0kGuAl8ZSIrhdFP7gDC2kTcrxmu56jcCNYrEuA8')
api = tweepy.API(auth, wait_on_rate_limit = True, 
                 wait_on_rate_limit_notify = True)


class MyStreamListener(tweepy.StreamListener):
    
    def __init__(self, time_limit = 600):
        self.start_time = time.time()
        self.limit = time_limit
        super(MyStreamListener, self).__init__()
    
    def on_status(self, status):
        if (time.time() - self.start_time) < self.limit:
            if hasattr(status, "extended_tweet"):
                text = status.extended_tweet["full_text"]
            else:
                text = status.text
            keys = ['covid', 'corona', 'pandemic', 'epidemic', 'reopen', 'quarantine',
                     'social distance', 'cough', 'fever', 'mask', 'virus', 'infect',
                     'contagious', 'stayhome']
            for term in keys:
                if term in text.lower():
                    out = " found"
                    print(out)
                    all_tweets.append(status)
                    break
            return True
        else:
            #self.output.close()
            return False
        
        
    def on_error(self, status_code):
        if status_code == 420:
            print('paused')
            return False
        

coords = pd.read_csv('US_State_Bounding_Boxes.csv')


all_states = {}
for i in range(coords.shape[0]):
    place = coords.loc[i, "NAME"]
    all_states[place] = []

for i in range(coords.shape[0]):
    box = [coords.loc[i, "xmin"], coords.loc[i, "ymin"], coords.loc[i, "xmax"],
           coords.loc[i, "ymax"]]
    place = coords.loc[i, "NAME"]
    duration = coords.loc[i, "duration"] * 60
    myStream = tweepy.Stream(auth = api.auth, listener = MyStreamListener(duration, place), 
                         tweet_mode = 'extended')
    myStream.filter(locations = box)

box = [min(coords["xmin"]), min(coords["ymin"]), max(coords["xmax"]), 
       max(coords["ymax"])]


all_tweets = []
myStream = tweepy.Stream(auth = api.auth, listener = MyStreamListener(36000), 
                         tweet_mode = 'extended')
myStream.filter(locations = box)

#two.place.bounding_box.coordinates[0]



''' Ignore Below
s1_json = json.dumps(status2._json)
s1_dict = json.loads(s1_json)



f = open("1120_covid.pkl", "wb")
pickle.dump(all_tweets, f)
f.close()

infile = open("0614_covid.pkl", "rb")
data0614 = pickle.load(infile)
infile.close()

infile = open("0615_covid.pkl", "rb")
data0615 = pickle.load(infile)
infile.close()

infile = open("0616_covid.pkl", "rb")
data0616 = pickle.load(infile)
infile.close()

infile = open("0617_covid.pkl", "rb")
data0617 = pickle.load(infile)
infile.close()

infile = open("0630_covid.pkl", "rb")
data0630 = pickle.load(infile)
infile.close()

num_tweet = {}
for i in data0615.keys():
    num_tweet[i] = len(data0614[i]) + len(data0615[i]) + len(data0616[i]) + \
    len(data0617[i]) + len(data0618[i])
    

test_s = data0617['Michigan'][0].extended_tweet['full_text']

broken_sent = nltk.sent_tokenize(test_s)

analyzer = SentimentIntensityAnalyzer()

for sentence in broken_sent:
    vs = analyzer.polarity_scores(sentence)
    print(vs)
    print("{:-<65} {}".format(sentence, str(vs)))
'''
